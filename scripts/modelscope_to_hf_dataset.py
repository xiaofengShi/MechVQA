#!/usr/bin/env python3
"""Mirror the public MechVQA dataset from ModelScope to Hugging Face.

Designed for a cloud runtime such as Google Colab, where both services are
reachable. The Hugging Face token is read from ``HF_TOKEN``, a Colab secret of
the same name, or a hidden prompt; it is never stored in this script.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from modelscope_hub import HubApi as ModelScopeHubApi


DEFAULT_MS_REPO = "xiaofengalg/MechVQA"
DEFAULT_HF_REPO = "XiaofengAlg/MechVQA"
EXPECTED_CHECKSUMS = 3383
KEY_FILES = (
    "README.md",
    "checksums.sha256",
    "data/train.jsonl",
    "data/val.jsonl",
    "audit/validation.json",
)
IGNORED_PREFIXES = (".cache/", ".modelscope/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelscope-repo", default=DEFAULT_MS_REPO)
    parser.add_argument("--huggingface-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/ms_dl/xiaofengalg/MechVQA"),
        help="Resumable ModelScope download directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use an existing release directory instead of downloading ModelScope.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source package without creating or uploading the HF repo.",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def resolve_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None

    if not token:
        token = getpass.getpass("Hugging Face write token: ").strip()
    if not token:
        raise RuntimeError("A Hugging Face write token is required.")
    return token


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_manifest(root: Path) -> int:
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {manifest}")

    checked = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"invalid checksum line {line_number}") from error
        relative = relative.lstrip("* ")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def package_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(IGNORED_PREFIXES):
            continue
        result.add(relative)
    return result


def download_source(args: argparse.Namespace) -> Path:
    if args.source_dir:
        source = args.source_dir.expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(source)
        return source

    args.work_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        ModelScopeHubApi().download_repo(
            args.modelscope_repo,
            repo_type="dataset",
            revision="master",
            local_dir=args.work_dir,
            max_workers=args.workers,
        )
    )


def verify_remote_key_files(
    api: HfApi, token: str, repo_id: str, source: Path
) -> None:
    for relative in KEY_FILES:
        remote = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=relative,
                repo_type="dataset",
                token=token,
                force_download=True,
            )
        )
        if sha256(remote) != sha256(source / relative):
            raise RuntimeError(f"remote key-file mismatch: {relative}")


def main() -> None:
    args = parse_args()
    print(f"Source: ModelScope dataset {args.modelscope_repo}", flush=True)
    print(f"Target: Hugging Face dataset {args.huggingface_repo}", flush=True)

    source = download_source(args)
    checked = verify_checksum_manifest(source)
    if checked != EXPECTED_CHECKSUMS:
        raise RuntimeError(
            f"expected {EXPECTED_CHECKSUMS} checksums, found {checked}"
        )
    source_files = package_files(source)
    print(
        f"Validated {checked} checksums across {len(source_files)} package files.",
        flush=True,
    )
    if args.dry_run:
        print("Dry run complete; nothing was uploaded.", flush=True)
        return

    token = resolve_hf_token()
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.huggingface_repo,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    api.update_repo_settings(
        repo_id=args.huggingface_repo,
        repo_type="dataset",
        private=False,
    )
    api.upload_large_folder(
        repo_id=args.huggingface_repo,
        repo_type="dataset",
        folder_path=source,
        private=False,
        ignore_patterns=[".cache/**", ".modelscope/**"],
        num_workers=args.workers,
        print_report=True,
        print_report_every=60,
    )

    remote_files = set(
        api.list_repo_files(args.huggingface_repo, repo_type="dataset")
    )
    missing = sorted(source_files - remote_files)
    if missing:
        raise RuntimeError(
            f"remote tree is missing {len(missing)} files: {missing[:5]}"
        )
    verify_remote_key_files(api, token, args.huggingface_repo, source)

    public_info = HfApi().dataset_info(args.huggingface_repo)
    if public_info.private:
        raise RuntimeError("the Hugging Face dataset is still private")

    print(
        "DONE "
        + json.dumps(
            {
                "source_files": len(source_files),
                "remote_files": len(remote_files),
                "checksums_verified": checked,
                "remote_key_files_verified": len(KEY_FILES),
                "public": True,
                "url": f"https://huggingface.co/datasets/{args.huggingface_repo}",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
