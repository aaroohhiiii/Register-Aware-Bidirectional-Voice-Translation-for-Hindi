"""deploy_hf.py — Upload the app to a Hugging Face Space (Docker SDK).

Copies only the files the server needs into a staging folder (no audio, slides,
zips, venvs or secrets), adds the Space README header, and uploads it in one
commit. Hugging Face then builds the Dockerfile.

Usage:
    tts-env\\Scripts\\hf auth login                       # once, paste a WRITE token
    tts-env\\Scripts\\python deploy_hf.py <username>/<space-name>
    tts-env\\Scripts\\python deploy_hf.py <username>/<space-name> --private
    tts-env\\Scripts\\python deploy_hf.py <username>/<space-name> --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Names skipped at any depth (shutil.ignore_patterns semantics).
IGNORE = [
    ".git", ".env", ".venv", "tts-env", ".cache", ".claude", "__pycache__", ".pytest_cache",
    "*.pyc", "*.pyo", "*.zip", "*.pptx", "*.wav", "*.mp3", ".DS_Store",
    "paper", "presentation", "project", "sample_audio", "demo", "tests", "tools",
    "clips", "clips_dirB", "tts_out", "asr_out", "e2e_out", "dirB_out", "ui_out",
    "gradio_cached_examples", "README.md", "DEPLOY.md",
]

SPACE_HEADER = """---
title: Register-Aware Hindi Voice Translation
emoji: 🗣️
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

"""


def stage(dest: Path) -> list[str]:
    shutil.copytree(ROOT, dest, ignore=shutil.ignore_patterns(*IGNORE), dirs_exist_ok=True)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (dest / "README.md").write_text(SPACE_HEADER + readme, encoding="utf-8")
    return sorted(str(p.relative_to(dest)).replace("\\", "/") for p in dest.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space_id", help="e.g. harshitak13/hindi-register-translation")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    parser.add_argument("--dry-run", action="store_true", help="list files that would be uploaded, then stop")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        size_kb = sum((Path(tmp) / f).stat().st_size for f in files) / 1024
        print(f"{len(files)} files, {size_kb:.0f} KB:")
        for f in files:
            print("  ", f)
        if args.dry_run:
            return

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.space_id, repo_type="space", space_sdk="docker",
                        private=args.private, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.space_id, repo_type="space",
                          commit_message="Deploy app")
        print(f"\nUploaded. Build logs: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
