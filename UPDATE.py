#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update từ GitHub - tải ZIP và giải nén. Không cần Git."""

import os, shutil, zipfile, urllib.request, tempfile
from pathlib import Path

TOOL_DIR = Path(__file__).parent
ZIP_URL = "https://github.com/nguyenvantuong161978-dotcom/master/archive/refs/heads/main.zip"

SKIP = ["progress.json", "queue_tracker.json", "timing_log.json", "run_log.txt",
        "github_token.txt", "thumb/creds.json", "thumb/thumbnails/", "thumb/assets/",
        "_designer_uploads/", "images/"]


def main():
    print("=" * 50)
    print("  UPDATE FROM GITHUB")
    print("=" * 50)

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "repo.zip")

    try:
        print("  Downloading...")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        print(f"  Downloaded: {os.path.getsize(zip_path) / 1024 / 1024:.1f} MB")

        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        src = [f for f in Path(temp_dir).iterdir() if f.is_dir() and "master" in f.name][0]

        copied = 0
        for item in src.rglob("*"):
            if not item.is_file():
                continue
            rel = str(item.relative_to(src)).replace("\\", "/")
            if any(rel == s or rel.startswith(s) for s in SKIP):
                continue
            dst = TOOL_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(item, dst)
                copied += 1
            except:
                pass

        print(f"  Updated: {copied} files")
        print()
        print("  UPDATE THANH CONG!")

    except Exception as e:
        print(f"  LOI: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    input("\nNhan Enter de dong...")


if __name__ == "__main__":
    main()
