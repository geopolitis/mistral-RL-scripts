#!/usr/bin/env python3
"""Upload a local model/adaptor folder to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub.\n"
        "Install it with one of:\n"
        "  pip install huggingface_hub\n"
        "  uv pip install --python .venv/bin/python huggingface_hub\n"
        "Then rerun this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local training artifacts to Hugging Face Hub")
    parser.add_argument(
        "--local-path",
        type=str,
        required=True,
        help="Local folder to upload (for example outputs/mistral-grpo or outputs/mistral-sft)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target HF repo id, e.g. your-user/ministral-grpo-lora",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repo type (default: model)",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist",
    )
    visibility.add_argument(
        "--public",
        action="store_true",
        help="Create repo as public if it does not exist",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model artifacts",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch/revision to upload to (default: main)",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional allow patterns passed to upload_folder",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*.tmp", "*.log", "__pycache__/*", "*.pyc"],
        help="Optional ignore patterns passed to upload_folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_path = Path(args.local_path).expanduser().resolve()
    if not local_path.exists() or not local_path.is_dir():
        raise FileNotFoundError(f"--local-path must be an existing directory: {local_path}")

    token = args.token.strip() if args.token else None
    if not token:
        raise ValueError("HF token missing. Set HF_TOKEN or pass --token.")

    api = HfApi(token=token)

    print(f"[hf] ensuring repo exists: {args.repo_id} ({args.repo_type})")
    private = False if args.public else args.private
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=private,
        exist_ok=True,
    )

    print(f"[hf] uploading folder: {local_path}")
    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(local_path),
        commit_message=args.commit_message,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )

    print(f"[hf] upload complete: {commit_info.commit_url}")
    print(f"[hf] repo: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
