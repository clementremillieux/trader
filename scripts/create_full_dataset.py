from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "datasets" / "reports"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _parse_csv(value: str | List[str], cast) -> List[Any]:
    if isinstance(value, list):
        return [cast(v) for v in value]
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return [cast(x) for x in items]


def _build_drive_service(
    service_account_path: Optional[Path],
    oauth_client_secrets: Optional[Path],
    oauth_token_path: Optional[Path],
):
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - dépendance optionnelle
        raise RuntimeError(
            "google-api-python-client n'est pas installé. Exécutez `poetry install` "
            "après avoir ajouté la dépendance ou retirez l'option Drive."
        ) from exc

    credentials = None
    if service_account_path is not None:
        credentials = service_account.Credentials.from_service_account_file(
            str(service_account_path), scopes=DRIVE_SCOPES
        )
    elif oauth_client_secrets is not None:
        token_path = oauth_token_path or (REPO_ROOT / "drive_token.json")
        creds = None
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_path), scopes=DRIVE_SCOPES
                )
            except Exception:
                creds = None
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if creds is None or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(oauth_client_secrets), scopes=DRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())
        credentials = creds
    else:  # tentative avec les identifiants par défaut (ADC)
        try:
            import google.auth  # type: ignore

            credentials, _ = google.auth.default(scopes=DRIVE_SCOPES)
        except Exception as exc:  # pragma: no cover - logging utilisateur
            raise RuntimeError(
                "Aucun identifiant Google Drive disponible. Fournissez "
                "--drive-service-account, configurez OAuth ou "
                "GOOGLE_APPLICATION_CREDENTIALS."
            ) from exc

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _escape_drive_name(name: str) -> str:
    return name.replace("'", "\\'")


class DriveUploader:
    def __init__(self, service, chunk_size: int = 20 * 1024 * 1024) -> None:
        from googleapiclient.http import MediaFileUpload

        self._service = service
        self._MediaFileUpload = MediaFileUpload
        self._folder_cache: Dict[tuple[str, str], str] = {}
        self._chunk_size = max(256 * 1024, (chunk_size // (256 * 1024)) * (256 * 1024))

    def ensure_folder(self, parent_id: str, folder_name: str) -> str:
        key = (parent_id, folder_name)
        if key in self._folder_cache:
            return self._folder_cache[key]

        query = (
            f"name = '{_escape_drive_name(folder_name)}' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            f"and '{parent_id}' in parents and trashed = false"
        )
        response = (
            self._service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name)",
                pageSize=1,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        if files:
            folder_id = files[0]["id"]
        else:
            metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            folder_id = (
                self._service.files()
                .create(body=metadata, fields="id", supportsAllDrives=True)
                .execute()["id"]
            )
        self._folder_cache[key] = folder_id
        return folder_id

    def upload_file(self, local_path: Path, parent_id: str) -> str:
        mime_type, _ = mimetypes.guess_type(local_path.name)
        media = self._MediaFileUpload(
            str(local_path),
            mimetype=mime_type or "application/octet-stream",
            resumable=True,
            chunksize=self._chunk_size,
        )
        metadata = {"name": local_path.name, "parents": [parent_id]}
        request = self._service.files().create(
            body=metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status is not None:
                progress = int(status.progress() * 100)
                print(f"[Drive] Téléversement {local_path.name}: {progress}%")
        return response["id"]


def _upload_directory(
    uploader: DriveUploader,
    local_root: Path,
    remote_root_id: str,
    remove_local: bool = False,
) -> None:
    local_root = local_root.resolve()
    for file_path in sorted(local_root.rglob("*")):
        if not file_path.is_file():
            continue
        rel_parts = file_path.relative_to(local_root).parts[:-1]
        parent_id = remote_root_id
        for part in rel_parts:
            parent_id = uploader.ensure_folder(parent_id, part)
        uploader.upload_file(file_path, parent_id)
        if remove_local:
            file_path.unlink()

    if remove_local:
        shutil.rmtree(local_root, ignore_errors=True)


def _stream_upload_directory(
    uploader: DriveUploader,
    local_root: Path,
    remote_root_id: str,
    remove_local: bool,
    process: subprocess.Popen,
    poll_interval: float = 15.0,
    stable_delay: float = 5.0,
) -> None:
    local_root = local_root.resolve()
    uploaded: Set[Path] = set()
    seen_sizes: Dict[Path, Tuple[int, float]] = {}

    def _collect_files() -> List[Path]:
        if not local_root.exists():
            return []
        return [p for p in local_root.rglob("*") if p.is_file()]

    def _upload_ready_files(force: bool = False) -> int:
        uploaded_now = 0
        current_files = _collect_files()
        existing_paths = set(current_files)
        # Nettoyage des entrées obsolètes
        for stale_path in list(seen_sizes):
            if stale_path not in existing_paths:
                seen_sizes.pop(stale_path, None)

        for file_path in current_files:
            if file_path in uploaded:
                continue

            try:
                stat = file_path.stat()
            except FileNotFoundError:
                continue

            now = time.monotonic()
            size = stat.st_size
            previous = seen_sizes.get(file_path)

            ready = False
            if previous is not None:
                prev_size, prev_time = previous
                if size == prev_size and (force or now - prev_time >= stable_delay):
                    ready = True

            seen_sizes[file_path] = (size, now)

            if not ready:
                continue

            rel_parts = file_path.relative_to(local_root).parts[:-1]
            parent_id = remote_root_id
            for part in rel_parts:
                parent_id = uploader.ensure_folder(parent_id, part)

            uploader.upload_file(file_path, parent_id)
            if remove_local:
                try:
                    file_path.unlink()
                except FileNotFoundError:
                    pass
            uploaded.add(file_path)
            seen_sizes.pop(file_path, None)
            uploaded_now += 1

        return uploaded_now

    # Première passe pour enregistrer les tailles initiales
    _collect_files()

    while True:
        _upload_ready_files()
        retcode = process.poll()
        if retcode is not None:
            # Processus terminé: on force quelques passes supplémentaires pour tout vider
            flush_start = time.monotonic()
            while True:
                uploaded_this_round = _upload_ready_files(force=True)
                remaining = [
                    p for p in _collect_files() if p not in uploaded and p.exists()
                ]
                if not remaining:
                    break
                # Évite les boucles infinies si certains fichiers restent verrous
                if time.monotonic() - flush_start > 300:
                    break
                if uploaded_this_round == 0:
                    time.sleep(min(poll_interval, 5.0))
            break

        time.sleep(poll_interval)

    if remove_local and local_root.exists():
        shutil.rmtree(local_root, ignore_errors=True)


def _run_dataset_with_streaming_upload(
    cmd: List[str],
    local_root: Path,
    uploader: DriveUploader,
    remote_root_id: str,
    remove_local: bool,
) -> None:
    # Injecte une variable d'environnement pour indiquer au processus enfant
    # qu'il est exécuté en mode streaming : il ne doit pas supprimer les shards
    env = os.environ.copy()
    env["STREAMING_UPLOAD_CHILD"] = "1"
    # Passe l'ID de dossier parent Drive au processus enfant pour logs/usage éventuelle
    env["DRIVE_PARENT_ID"] = str(remote_root_id)
    print("[Parent] Lancement du processus enfant avec STREAMING_UPLOAD_CHILD=1")
    process = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
    try:
        _stream_upload_directory(
            uploader,
            local_root,
            remote_root_id,
            remove_local,
            process,
        )
    finally:
        retcode = process.wait()

    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd)


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
        "--output-dir",
        str(args.full_output_dir),
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
    if args.full_disable_auto_balance:
        cmd.append("--no-auto-balance")
    else:
        cmd.extend(
            [
                "--balance-max-multiplier",
                str(args.full_balance_max_multiplier),
            ]
        )
    if args.full_disable_profit_boost:
        cmd.extend(["--profit-boost-extra-fraction", "0"])
    else:
        cmd.extend(
            [
                "--profit-boost-quantile",
                str(args.full_profit_boost_quantile),
                "--profit-boost-extra-fraction",
                str(args.full_profit_boost_extra_fraction),
                "--profit-boost-max-multiplier",
                str(args.full_profit_boost_max_multiplier),
            ]
        )
        if args.full_profit_boost_min_gain > 0:
            cmd.extend(
                [
                    "--profit-boost-min-gain",
                    str(args.full_profit_boost_min_gain),
                ]
            )
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
    parser.add_argument("--bin-thresholds", default="0.8,1.0,1.2,1.5")
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
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=None,
        help="Chemin d'un fichier de résumé de sweep existant à utiliser avec --skip-sweep.",
    )
    parser.add_argument("--skip-sweep", action="store_true")

    parser.add_argument("--full-out-prefix", default="full_dataset")
    parser.add_argument("--full-tickers-per-batch", type=int, default=10)
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
    parser.add_argument("--full-output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--full-dataset-version", default="crypto_v3_dataset_best")
    parser.add_argument(
        "--full-quality-report",
        type=Path,
        default=Path("datasets/quality_report_full.json"),
    )
    parser.add_argument("--full-disable-auto-balance", action="store_true")
    parser.add_argument(
        "--full-balance-max-multiplier",
        type=float,
        default=1.6,
        help="Facteur max d'augmentation du dataset via équilibrage.",
    )
    parser.add_argument("--full-disable-profit-boost", action="store_true")
    parser.add_argument(
        "--full-profit-boost-quantile",
        type=float,
        default=0.8,
        help="Quantile utilisés pour choisir les trades à dupliquer.",
    )
    parser.add_argument(
        "--full-profit-boost-extra-fraction",
        type=float,
        default=0.15,
        help="Fraction max du dataset ajoutée via duplication rentable.",
    )
    parser.add_argument(
        "--full-profit-boost-max-multiplier",
        type=float,
        default=1.5,
        help="Multiplicateur global max après duplication rentable.",
    )
    parser.add_argument(
        "--full-profit-boost-min-gain",
        type=float,
        default=0.0,
        help="Gain directionnel minimum pour la duplication rentable.",
    )
    parser.add_argument("--drive-folder-id", default=None)
    parser.add_argument("--drive-service-account", type=Path, default=None)
    parser.add_argument(
        "--drive-oauth-client-secrets",
        type=Path,
        default=None,
        help="Chemin vers le fichier client OAuth (authentification utilisateur)",
    )
    parser.add_argument(
        "--drive-oauth-token-path",
        type=Path,
        default=None,
        help="Chemin d'enregistrement du token OAuth (défaut: drive_token.json)",
    )
    parser.add_argument("--drive-subfolder-name", default=None)
    parser.add_argument("--drive-remove-local", action="store_true")
    parser.add_argument(
        "--drive-chunk-size",
        type=int,
        default=20 * 1024 * 1024,
        help="Taille des chunks (en octets) pour l'upload Drive (multiple de 256k).",
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
    # Préparation des chemins de sortie
    args.full_output_dir = args.full_output_dir.expanduser()
    args.full_quality_report = args.full_quality_report.expanduser()
    drive_service_account: Optional[Path] = None
    if args.drive_service_account is not None:
        drive_service_account = args.drive_service_account.expanduser().resolve()

    drive_oauth_client_secrets: Optional[Path] = None
    if args.drive_oauth_client_secrets is not None:
        drive_oauth_client_secrets = (
            args.drive_oauth_client_secrets.expanduser().resolve()
        )

    drive_oauth_token_path: Optional[Path] = None
    if args.drive_oauth_token_path is not None:
        drive_oauth_token_path = args.drive_oauth_token_path.expanduser()
    reports_dir = _ensure_reports_dir()

    upload_to_drive = bool(args.drive_folder_id)
    run_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    if upload_to_drive:
        run_dir_name = f"{args.full_out_prefix}_{run_timestamp}"
        args.full_output_dir = args.full_output_dir / run_dir_name
        if not args.drive_subfolder_name:
            args.drive_subfolder_name = run_dir_name

    args.full_output_dir.mkdir(parents=True, exist_ok=True)
    if upload_to_drive:
        args.full_quality_report = args.full_output_dir / args.full_quality_report.name

    drive_uploader: Optional[DriveUploader] = None
    if upload_to_drive and not args.dry_run:
        service = _build_drive_service(
            drive_service_account,
            drive_oauth_client_secrets,
            drive_oauth_token_path,
        )
        drive_uploader = DriveUploader(service, chunk_size=args.drive_chunk_size)

    if upload_to_drive and args.dry_run:
        print("[Dry-run] Upload Google Drive demandé mais non exécuté.")

    summary_path = args.sweep_summary
    if args.skip_sweep:
        # Si --skip-sweep, summary_path doit être fourni et exister
        if summary_path is None:
            raise FileNotFoundError(
                "Vous devez fournir --sweep-summary <fichier> avec --skip-sweep pour indiquer la config à utiliser."
            )
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Résumé de sweep introuvable ({summary_path}). Impossible de sauter l'étape de sweep."
            )
        best_cfg = _load_best_config(summary_path)
    else:
        if summary_path is None:
            summary_path = (
                reports_dir / f"sweep_summary_{datetime.utcnow():%Y%m%dT%H%M%S}.json"
            )
        env = os.environ.copy()
        env["SWEEP_MAX_TRAIN_SAMPLES"] = str(args.sweep_max_train_samples)
        env["SWEEP_MAX_VAL_SAMPLES"] = str(args.sweep_max_val_samples)
        sweeper_cmd = _build_sweeper_command(args, summary_path)
        run_command(sweeper_cmd, env=env, dry_run=args.dry_run)
        best_cfg = _load_best_config(summary_path)

    print("Meilleure configuration trouvée:")
    print(json.dumps(best_cfg, indent=2, ensure_ascii=False))

    dataset_cmd = _build_dataset_command(best_cfg, args)

    if upload_to_drive and not args.dry_run and drive_uploader is not None:
        assert args.drive_subfolder_name is not None
        remote_root = args.drive_folder_id
        assert remote_root is not None
        remote_subfolder_id = drive_uploader.ensure_folder(
            remote_root, args.drive_subfolder_name
        )
        _run_dataset_with_streaming_upload(
            dataset_cmd,
            args.full_output_dir,
            drive_uploader,
            remote_subfolder_id,
            remove_local=args.drive_remove_local,
        )
        print(
            "Upload Google Drive terminé → dossier:",
            f"https://drive.google.com/drive/folders/{remote_subfolder_id}",
        )
    else:
        run_command(dataset_cmd, dry_run=args.dry_run)
        if upload_to_drive and not args.dry_run and drive_uploader is not None:
            remote_root = args.drive_folder_id
            assert remote_root is not None
            remote_subfolder_id = drive_uploader.ensure_folder(
                remote_root, args.drive_subfolder_name or args.full_out_prefix
            )
            _upload_directory(
                drive_uploader,
                args.full_output_dir,
                remote_subfolder_id,
                remove_local=args.drive_remove_local,
            )
            print(
                "Upload Google Drive terminé → dossier:",
                f"https://drive.google.com/drive/folders/{remote_subfolder_id}",
            )


if __name__ == "__main__":
    main()
