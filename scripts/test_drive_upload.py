from __future__ import annotations

import argparse
import mimetypes
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


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
            "Les bibliothèques Google Drive ne sont pas installées. "
            "Lancez `poetry install` pour les ajouter."
        ) from exc

    credentials = None
    if service_account_path is not None:
        credentials = service_account.Credentials.from_service_account_file(
            str(service_account_path), scopes=DRIVE_SCOPES
        )
    elif oauth_client_secrets is not None:
        token_path = oauth_token_path or Path("drive_token.json")
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
    else:
        try:
            import google.auth  # type: ignore

            credentials, _ = google.auth.default(scopes=DRIVE_SCOPES)
        except Exception as exc:
            raise RuntimeError(
                "Impossible de récupérer des identifiants Google Drive. "
                "Fournissez --drive-service-account, configurez OAuth ou "
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test d'upload d'un petit fichier sur Google Drive"
    )
    parser.add_argument(
        "--drive-folder-id",
        required=True,
        help="Identifiant du dossier Drive cible",
    )
    parser.add_argument(
        "--drive-service-account",
        type=Path,
        default=None,
        help=(
            "Chemin vers un JSON de compte de service (facultatif si "
            "GOOGLE_APPLICATION_CREDENTIALS est défini)"
        ),
    )
    parser.add_argument(
        "--oauth-client-secrets",
        type=Path,
        default=None,
        help="Fichier client OAuth (si vous préférez l'authentification utilisateur)",
    )
    parser.add_argument(
        "--oauth-token-path",
        type=Path,
        default=None,
        help="Fichier token OAuth à réutiliser (défaut: drive_token.json)",
    )
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Nom d'un sous-dossier à créer sous le dossier Drive cible",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Fichier local à envoyer (par défaut un fichier texte temporaire sera généré)",
    )
    parser.add_argument(
        "--content",
        default="Hello from trader-auto!",
        help="Contenu à écrire si le fichier est généré automatiquement",
    )
    parser.add_argument(
        "--filename",
        default="drive_test.txt",
        help="Nom du fichier généré lorsqu'aucun fichier n'est fourni",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4 * 1024 * 1024,
        help="Taille des chunks pour l'upload (multiple de 256k)",
    )
    parser.add_argument(
        "--keep-temp-file",
        action="store_true",
        help="Ne pas supprimer le fichier temporaire généré",
    )
    args = parser.parse_args()

    service_account_path: Optional[Path] = None
    if args.drive_service_account is not None:
        service_account_path = args.drive_service_account.expanduser().resolve()

    oauth_client_secrets: Optional[Path] = None
    if args.oauth_client_secrets is not None:
        oauth_client_secrets = args.oauth_client_secrets.expanduser().resolve()

    oauth_token_path: Optional[Path] = None
    if args.oauth_token_path is not None:
        oauth_token_path = args.oauth_token_path.expanduser()

    service = _build_drive_service(
        service_account_path,
        oauth_client_secrets,
        oauth_token_path,
    )
    uploader = DriveUploader(service, chunk_size=args.chunk_size)

    local_file: Path
    temp_file = None
    if args.file is not None:
        local_file = args.file.expanduser().resolve()
        if not local_file.exists():
            raise FileNotFoundError(f"Fichier spécifié introuvable: {local_file}")
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="drive_test_"))
        local_file = tmp_dir / args.filename
        local_file.write_text(args.content, encoding="utf-8")
        temp_file = tmp_dir
        print(f"Fichier temporaire créé: {local_file}")

    parent_id = args.drive_folder_id
    if args.subfolder:
        parent_id = uploader.ensure_folder(args.drive_folder_id, args.subfolder)

    file_id = uploader.upload_file(local_file, parent_id)
    print("Upload terminé → ID:", file_id)
    print("URL: https://drive.google.com/file/d/" + file_id)

    if temp_file is not None and not args.keep_temp_file:
        shutil.rmtree(temp_file, ignore_errors=True)


if __name__ == "__main__":
    main()
