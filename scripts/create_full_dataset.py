from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "datasets" / "reports"


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _parse_csv(value: str | List[str], cast) -> List[Any]:
    if isinstance(value, list):
        return [cast(v) for v in value]
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return [cast(x) for x in items]


def _build_sweeper_command(args: argparse.Namespace, summary_path: Path) -> List[str]:
    script = REPO_ROOT / "scripts" / "config_corr_sweeper.py"
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--intervals",
        ",".join(args.intervals),
        "--windows",
        ",".join(str(w) for w in args.windows),
        "--strides",
        ",".join(str(s) for s in args.strides),
        "--bin-hs",
        ",".join(str(h) for h in args.bin_horizons),
        "--bin-thrs",
        ",".join(str(thr) for thr in args.bin_thresholds),
        "--tickers-per-batch",
        str(args.sweep_tickers_per_batch),
        "--max-batches",
        str(args.sweep_max_batches),
        "--min-train-windows",
        str(args.sweep_min_train_windows),
        "--min-val-windows",
        str(args.sweep_min_val_windows),
        "--score-target",
        args.score_target,
        "--score-metric",
        args.score_metric,
        "--lags",
        ",".join(str(lag) for lag in args.lags),
        "--report",
        str(summary_path),
    ]
    if not args.time_aware_periods:
        cmd.append("--no-time-aware")
    if args.disable_cache:
        cmd.append("--disable-cache")
    if args.tickers_allow:
        cmd.extend(["--tickers-allow", args.tickers_allow])
    if args.no_generate:
        cmd.append("--no-generate")
    if args.cache_ttl_hours is not None:
        cmd.extend(["--cache-ttl", str(args.cache_ttl_hours)])
    if args.extra_sweeper_args:
        cmd.extend(args.extra_sweeper_args)
    return cmd


def _load_best_config(summary_path: Path) -> Dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    if (
        isinstance(summary, dict)
        and "best" in summary
        and isinstance(summary["best"], dict)
    ):
        best_entry = summary["best"]
        if "exp" in best_entry and isinstance(best_entry["exp"], dict):
            return best_entry["exp"]
        return best_entry
    raise ValueError(f"Fichier de sweep invalide: {summary_path}")


def _build_dataset_command(
    best_cfg: Dict[str, Any], args: argparse.Namespace
) -> List[str]:
    script = REPO_ROOT / "dataset_creator_binance_buy.py"
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--interval",
        best_cfg["interval"],
        "--window",
        str(best_cfg["window"]),
        "--window-stride",
        str(best_cfg["stride"]),
        "--bin-label-h",
        str(best_cfg["bin_h"]),
        "--bin-label-thr-atr",
        str(best_cfg["bin_thr_atr"]),
        "--tickers-per-batch",
        str(args.full_tickers_per_batch),
        "--max-batches",
        str(args.full_max_batches if args.full_max_batches is not None else 0),
        "--start-batch",
        str(args.full_start_batch),
        "--split-hours",
        str(args.full_split_hours),
        "--purge-hours",
        str(args.full_purge_hours),
        "--min-train-windows",
        str(args.full_min_train_windows),
        "--min-val-windows",
        str(args.full_min_val_windows),
        "--max-candles",
        str(args.full_max_candles),
        "--out-prefix",
        args.full_out_prefix,
        "--dataset-version",
        args.full_dataset_version,
        "--quality-report",
        str(args.full_quality_report),
    ]
    if best_cfg.get("time_aware", False) or args.time_aware_periods:
        cmd.append("--time-aware-periods")
    if not args.full_produce_binary_cls:
        cmd.append("--no-produce-binary-cls")
    if args.full_disable_cache:
        cmd.append("--disable-cache")
    if args.full_resume:
        cmd.append("--resume")
    if args.full_tickers_allow:
        cmd.extend(["--tickers-allow", args.full_tickers_allow])
    if args.full_cache_ttl is not None:
        cmd.extend(["--cache-ttl", str(args.full_cache_ttl)])
    if args.full_max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(args.full_max_train_samples)])
    if args.full_max_val_samples is not None:
        cmd.extend(["--max-val-samples", str(args.full_max_val_samples)])
    if args.extra_dataset_args:
        cmd.extend(args.extra_dataset_args)
    return cmd


def run_command(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> None:
    print("→", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Balaye les configurations de corrélation puis génère le dataset complet "
            "sur l'ensemble des tickers avec la meilleure configuration."
        )
    )
    parser.add_argument("--intervals", default="1h")
    parser.add_argument("--windows", default="256,300")
    parser.add_argument("--strides", default="6")
    parser.add_argument("--bin-horizons", default="18,24,30")
    parser.add_argument("--bin-thresholds", default="0.8,1.0,1.2")
    parser.add_argument("--score-target", default="y_bin")
    parser.add_argument(
        "--score-metric",
        default="mi",
        choices=["mi", "pearson", "spearman"],
    )
    parser.add_argument("--lags", default="0,6,12,24,48")
    parser.add_argument("--time-aware-periods", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--tickers-allow", default=None)
    parser.add_argument("--cache-ttl-hours", type=int, default=None)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--sweep-tickers-per-batch", type=int, default=8)
    parser.add_argument("--sweep-max-batches", type=int, default=1)
    parser.add_argument("--sweep-min-train-windows", type=int, default=300)
    parser.add_argument("--sweep-min-val-windows", type=int, default=100)
    parser.add_argument("--sweep-max-train-samples", type=int, default=20_000)
    parser.add_argument("--sweep-max-val-samples", type=int, default=8_000)
    parser.add_argument("--extra-sweeper-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--sweep-summary", type=Path, default=None)
    parser.add_argument("--skip-sweep", action="store_true")

    parser.add_argument("--full-out-prefix", default="full_dataset")
    parser.add_argument("--full-tickers-per-batch", type=int, default=20)
    parser.add_argument("--full-max-batches", type=int, default=None)
    parser.add_argument("--full-start-batch", type=int, default=0)
    parser.add_argument("--full-min-train-windows", type=int, default=800)
    parser.add_argument("--full-min-val-windows", type=int, default=200)
    parser.add_argument("--full-max-candles", type=int, default=150_000)
    parser.add_argument("--full-split-hours", type=int, default=1000)
    parser.add_argument("--full-purge-hours", type=int, default=12)
    parser.add_argument("--full-cache-ttl", type=int, default=None)
    parser.add_argument("--full-disable-cache", action="store_true")
    parser.add_argument("--full-resume", action="store_true")
    parser.add_argument("--full-produce-binary-cls", action="store_true", default=True)
    parser.add_argument("--full-tickers-allow", default=None)
    parser.add_argument("--full-max-train-samples", type=int, default=None)
    parser.add_argument("--full-max-val-samples", type=int, default=None)
    parser.add_argument("--full-dataset-version", default="crypto_v3_dataset_best")
    parser.add_argument(
        "--full-quality-report",
        type=Path,
        default=Path("datasets/quality_report_full.json"),
    )
    parser.add_argument("--extra-dataset-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    args.intervals = _parse_csv(args.intervals, str)
    args.windows = _parse_csv(args.windows, int)
    args.strides = _parse_csv(args.strides, int)
    args.bin_horizons = _parse_csv(args.bin_horizons, int)
    args.bin_thresholds = _parse_csv(args.bin_thresholds, float)
    args.lags = _parse_csv(args.lags, int)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    reports_dir = _ensure_reports_dir()

    summary_path = args.sweep_summary
    if summary_path is None:
        summary_path = (
            reports_dir / f"sweep_summary_{datetime.utcnow():%Y%m%dT%H%M%S}.json"
        )

    if args.skip_sweep:
        if not summary_path.exists():
            raise FileNotFoundError(
                "Résumé de sweep introuvable"
                f" ({summary_path}). Impossible de sauter l'étape de sweep."
            )
        best_cfg = _load_best_config(summary_path)
    else:
        env = os.environ.copy()
        env["SWEEP_MAX_TRAIN_SAMPLES"] = str(args.sweep_max_train_samples)
        env["SWEEP_MAX_VAL_SAMPLES"] = str(args.sweep_max_val_samples)
        sweeper_cmd = _build_sweeper_command(args, summary_path)
        run_command(sweeper_cmd, env=env, dry_run=args.dry_run)
        best_cfg = _load_best_config(summary_path)

    print("Meilleure configuration trouvée:")
    print(json.dumps(best_cfg, indent=2, ensure_ascii=False))

    dataset_cmd = _build_dataset_command(best_cfg, args)
    run_command(dataset_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
