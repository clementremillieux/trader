from __future__ import annotations

import argparse
import itertools
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ce script orchestre des expériences de génération + évaluation, et sélectionne
# la configuration qui maximise une métrique de dépendance (corrélation de Pearson,
# Spearman approx., ou information mutuelle) entre les features et la cible choisie
# (y_bin, y_cls, y_reg). Il s'appuie sur:
#   - dataset_creator_binance_buy.py (générateur)
#   - scripts/dataset_quick_eval.py (évaluateur avec --json-out)
# Les sorties JSON sont sauvegardées dans datasets/reports/

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
REPORTS_DIR = DATASETS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Experiment:
    interval: str
    window: int
    stride: int
    bin_h: int
    bin_thr_atr: float
    time_aware: bool = True
    out_prefix: Optional[str] = None  # si None → auto

    def prefix(self) -> str:
        if self.out_prefix:
            return self.out_prefix
        tap = "tap1" if self.time_aware else "tap0"
        return (
            f"sweep_{self.interval}_w{self.window}_s{self.stride}_"
            f"h{self.bin_h}_thr{self.bin_thr_atr:g}_{tap}"
        )


@dataclass
class SweepConfig:
    # grille par défaut
    intervals: List[str] = None  # type: ignore[assignment]
    windows: List[int] = None  # type: ignore[assignment]
    strides: List[int] = None  # type: ignore[assignment]
    bin_hs: List[int] = None  # type: ignore[assignment]
    bin_thrs: List[float] = None  # type: ignore[assignment]
    time_aware_periods: bool = True

    # génération
    tickers_per_batch: int = 12
    max_batches: int = 1
    min_train_windows: int = 300
    min_val_windows: int = 100
    tickers_allow: Optional[str] = None  # CSV optionnel
    cache_ttl_hours: Optional[int] = None

    # évaluation
    score_target: str = "y_bin"  # y_bin|y_cls|y_reg
    score_metric: str = "pearson"  # pearson|spearman|mi
    topk: int = 8
    lags: str = "0,6,12,24,48,96"

    # contrôle
    generate: bool = True  # lancer la génération avant l'évaluation
    disable_cache: bool = False

    def __post_init__(self):
        if self.intervals is None:
            self.intervals = ["1h", "4h"]
        if self.windows is None:
            self.windows = [256, 300]
        if self.strides is None:
            self.strides = [6, 12]
        if self.bin_hs is None:
            self.bin_hs = [12, 24]
        if self.bin_thrs is None:
            self.bin_thrs = [0.8, 1.0, 1.2, 1.5]


def build_experiments(cfg: SweepConfig) -> List[Experiment]:
    exps: List[Experiment] = []
    for interval, window, stride, bin_h, thr in itertools.product(
        cfg.intervals, cfg.windows, cfg.strides, cfg.bin_hs, cfg.bin_thrs
    ):
        exps.append(
            Experiment(
                interval=interval,
                window=window,
                stride=stride,
                bin_h=bin_h,
                bin_thr_atr=float(thr),
                time_aware=cfg.time_aware_periods,
            )
        )
    return exps


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Exécute une commande et retourne (code, stdout, stderr)."""
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO_ROOT)
    )
    out_b, err_b = p.communicate()
    return (
        p.returncode,
        out_b.decode("utf-8", errors="replace"),
        err_b.decode("utf-8", errors="replace"),
    )


def generate_dataset(exp: Experiment, cfg: SweepConfig) -> None:
    """Lance la génération pour une config; outputs dans datasets/ avec prefix distinct."""
    script = REPO_ROOT / "dataset_creator_binance_buy.py"
    prefix = exp.prefix()
    args = [
        sys.executable,
        str(script),
        "--interval",
        exp.interval,
        "--window",
        str(exp.window),
        "--window-stride",
        str(exp.stride),
        "--start-batch",
        "0",
        "--bin-label-h",
        str(exp.bin_h),
        "--bin-label-thr-atr",
        str(exp.bin_thr_atr),
        "--out-prefix",
        prefix,
        "--tickers-per-batch",
        str(cfg.tickers_per_batch),
        "--max-batches",
        str(cfg.max_batches),
        "--min-train-windows",
        str(cfg.min_train_windows),
        "--min-val-windows",
        str(cfg.min_val_windows),
        "--split-hours",
        "1000",
        "--purge-hours",
        "12",
    ]
    # Pass-through caps to reduce dataset size if defined via env
    max_tr = os.getenv("SWEEP_MAX_TRAIN_SAMPLES")
    max_va = os.getenv("SWEEP_MAX_VAL_SAMPLES")
    if max_tr:
        args += ["--max-train-samples", max_tr]
    if max_va:
        args += ["--max-val-samples", max_va]
    if cfg.tickers_allow:
        args += ["--tickers-allow", cfg.tickers_allow]
    if cfg.disable_cache:
        args += ["--disable-cache"]
    if cfg.cache_ttl_hours is not None:
        args += ["--cache-ttl", str(cfg.cache_ttl_hours)]
    if exp.time_aware:
        args += ["--time-aware-periods"]

    code, out, err = run_cmd(args)
    if code != 0:
        print(f"[ERR] Génération KO ({prefix})\nSTDERR:\n{err}\nSTDOUT:\n{out}")
        raise SystemExit(code)


def evaluate_dataset(prefix: str, lags: str, json_out: Path) -> Dict[str, Any]:
    script = REPO_ROOT / "scripts" / "dataset_quick_eval.py"
    args = [
        sys.executable,
        str(script),
        "--prefix",
        prefix,
        "--lags",
        lags,
        "--json-out",
        str(json_out),
    ]
    code, out, err = run_cmd(args)
    if code != 0:
        print(f"[ERR] Évaluation KO ({prefix})\nSTDERR:\n{err}\nSTDOUT:\n{out}")
        raise SystemExit(code)
    try:
        with open(json_out, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        print(f"[ERR] Lecture JSON impossible: {exc}")
        raise


def _glob_any(prefix: str, split: str) -> list[Path]:
    pattern = f"datasets/{prefix}_{split}_*.pickle"
    # On ignore les variantes *scaled* si elles existent
    return [Path(p) for p in glob.glob(pattern) if "scaled" not in p]


def score_experiment(
    report: Dict[str, Any], target: str, metric: str, topk: int
) -> float:
    """Calcule un score numérique à partir du rapport JSON.

    - metric=pearson|spearman: moyenne des |tops| topk
    - metric=mi: moyenne des tops MI topk
    """
    tops = report.get("top", {})
    if metric == "pearson":
        key = tops.get("pearson", {}).get(target)
        if not key:
            return float("nan")
        vals = [abs(float(x[2])) for x in key[:topk]]
        return float(sum(vals) / max(1, len(vals)))
    if metric == "spearman":
        key = tops.get("spearman", {}).get(target)
        if not key:
            return float("nan")
        vals = [abs(float(x[2])) for x in key[:topk]]
        return float(sum(vals) / max(1, len(vals)))
    if metric == "mi":
        key = tops.get("mi", {}).get(target)
        if not key:
            return float("nan")
        vals = [float(x[2]) for x in key[:topk]]
        return float(sum(vals) / max(1, len(vals)))
    return float("nan")


def parse_cli() -> Tuple[SweepConfig, List[Experiment]]:
    p = argparse.ArgumentParser(
        description=(
            "Balaye des configurations et sélectionne celle qui maximise la "
            "corrélation (ou MI) des features avec la cible."
        )
    )
    p.add_argument("--intervals", default="1h,4h")
    p.add_argument("--windows", default="256,300")
    p.add_argument("--strides", default="6,12")
    p.add_argument("--bin-hs", default="12,24")
    p.add_argument("--bin-thrs", default="0.5,1.0")
    p.add_argument("--no-time-aware", action="store_true")
    p.add_argument("--tickers-per-batch", type=int, default=12)
    p.add_argument("--max-batches", type=int, default=1)
    p.add_argument("--min-train-windows", type=int, default=300)
    p.add_argument("--min-val-windows", type=int, default=100)
    p.add_argument("--tickers-allow", default=None)
    p.add_argument("--disable-cache", action="store_true")
    p.add_argument("--cache-ttl", type=int, default=None)
    p.add_argument("--score-target", default="y_bin")
    p.add_argument("--score-metric", default="pearson")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--lags", default="0,6,12,24,48,96")
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--report", default=None, help="Fichier JSON récapitulatif")

    args = p.parse_args()

    def _parse_list(s: str, cast):
        return [cast(x) for x in s.split(".")]

    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    strides = [int(x) for x in args.strides.split(",") if x.strip()]
    bin_hs = [int(x) for x in args.bin_hs.split(",") if x.strip()]
    bin_thrs = [float(x) for x in args.bin_thrs.split(",") if x.strip()]

    cfg = SweepConfig(
        intervals=intervals,
        windows=windows,
        strides=strides,
        bin_hs=bin_hs,
        bin_thrs=bin_thrs,
        time_aware_periods=(not args.no_time_aware),
        tickers_per_batch=args.tickers_per_batch,
        max_batches=args.max_batches,
        min_train_windows=args.min_train_windows,
        min_val_windows=args.min_val_windows,
        tickers_allow=args.tickers_allow,
        cache_ttl_hours=args.cache_ttl,
        score_target=args.score_target,
        score_metric=args.score_metric,
        topk=args.topk,
        lags=args.lags,
        generate=(not args.no_generate),
        disable_cache=args.disable_cache,
    )

    exps = build_experiments(cfg)
    return cfg, exps


def main():
    cfg, exps = parse_cli()
    print(f"Nombre de configurations: {len(exps)}")

    results: List[Dict[str, Any]] = []
    for i, exp in enumerate(exps, start=1):
        print(f"[{i}/{len(exps)}] {exp.prefix()}")
        if cfg.generate:
            generate_dataset(exp, cfg)

        # Vérifier la présence des pickles avant d'évaluer
        missing: list[str] = []
        for split in ("train", "val"):
            files = _glob_any(exp.prefix(), split)
            if not files:
                missing.append(split)

        if missing:
            msg = (
                "  [WARN] Aucuns pickles trouvés pour "
                + ",".join(missing)
                + " → score=NaN, on passe."
            )
            print(msg)
            results.append(
                {
                    "exp": asdict(exp),
                    "score": float("nan"),
                    "report": None,
                    "baselines": {},
                    "missing": missing,
                }
            )
            continue

        # Évaluation et scoring
        json_path = REPORTS_DIR / f"{exp.prefix()}_eval.json"
        rep = evaluate_dataset(exp.prefix(), cfg.lags, json_path)
        sc = score_experiment(rep, cfg.score_target, cfg.score_metric, cfg.topk)
        results.append(
            {
                "exp": asdict(exp),
                "score": sc,
                "report": str(json_path),
                "baselines": rep.get("baselines", {}),
            }
        )
        print(f"  score={sc:.4f}  → report={json_path.name}")

    # Sélection
    # garder résultats dont le score n'est pas NaN
    valid_res = [
        r
        for r in results
        if not (isinstance(r["score"], float) and (r["score"] != r["score"]))
    ]
    if not valid_res:
        print("Aucun score valide; vérifiez les rapports JSON.")
        return
    best = (
        max(valid_res, key=lambda r: r["score"])
        if cfg.score_metric in ("pearson", "spearman", "mi")
        else valid_res[0]
    )
    print("\nMeilleure configuration:")
    print(json.dumps(best, indent=2))

    # Rapport récapitulatif
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": asdict(cfg),
        "results": results,
        "best": best,
    }
    # choix du nom de fichier
    out_name = f"sweep_summary_{int(time.time())}.json"
    final_path = REPORTS_DIR / out_name
    try:
        # si l'utilisateur a donné --report, l'utiliser
        cli_args = sys.argv[1:]
        if "--report" in cli_args:
            idx = cli_args.index("--report")
            if idx + 1 < len(cli_args):
                user_out = cli_args[idx + 1]
                if user_out:
                    final_path = (
                        REPORTS_DIR / user_out
                        if not os.path.isabs(user_out)
                        else Path(user_out)
                    )
        with open(final_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\nRésumé écrit → {final_path}")
    except OSError as exc:
        print(f"[WARN] Échec écriture résumé: {exc}")


if __name__ == "__main__":
    main()
