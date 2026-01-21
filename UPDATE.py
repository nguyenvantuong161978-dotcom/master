#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update từ GitHub - Không cần cài Git
Tải ZIP và giải nén để cập nhật code mới nhất.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import tempfile
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

GITHUB_REPO = "nguyenvantuong161978-dotcom/master"
GITHUB_BRANCH = "main"
ZIP_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"

# Thư mục hiện tại
TOOL_DIR = Path(__file__).parent

# Files cần giữ lại (không ghi đè)
KEEP_FILES = [
    "config/config.json",
    "config/creds.json",
    "config/github_token.txt",
]

# Thư mục cần giữ lại
KEEP_FOLDERS = [
    "PROJECTS",
]


# ============================================================================
# UPDATE FUNCTIONS
# ============================================================================

def download_zip(url, dest_path):
    """Download ZIP file từ URL."""
    print(f"  Downloading from GitHub...")
    print(f"  URL: {url[:60]}...")

    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  Downloaded: {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Giải nén ZIP file."""
    print(f"  Extracting ZIP...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)

        # Tìm thư mục được giải nén (thường là repo-branch)
        extracted_folders = [f for f in Path(extract_to).iterdir() if f.is_dir()]
        if extracted_folders:
            return extracted_folders[0]
        return None
    except Exception as e:
        print(f"  [ERROR] Extract failed: {e}")
        return None


def backup_keep_files():
    """Backup các file cần giữ."""
    backups = {}

    for rel_path in KEEP_FILES:
        src = TOOL_DIR / rel_path
        if src.exists():
            backups[rel_path] = src.read_bytes()
            print(f"  Backed up: {rel_path}")

    return backups


def restore_keep_files(backups):
    """Restore các file đã backup."""
    for rel_path, content in backups.items():
        dst = TOOL_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
        print(f"  Restored: {rel_path}")


def copy_new_files(src_dir, dst_dir):
    """Copy files từ thư mục mới sang thư mục hiện tại."""
    copied = 0
    skipped = 0

    for item in src_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src_dir)
            rel_str = str(rel_path).replace("\\", "/")

            # Skip files cần giữ
            if rel_str in KEEP_FILES:
                skipped += 1
                continue

            # Skip folders cần giữ
            skip = False
            for keep_folder in KEEP_FOLDERS:
                if rel_str.startswith(keep_folder + "/") or rel_str == keep_folder:
                    skip = True
                    break
            if skip:
                skipped += 1
                continue

            # Copy file
            dst = dst_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(item, dst)
                copied += 1
            except Exception as e:
                print(f"  [WARN] Cannot copy {rel_path}: {e}")

    return copied, skipped


def main():
    print("=" * 55)
    print("  UPDATE FROM GITHUB (No Git Required)")
    print("=" * 55)
    print(f"  Repo:   {GITHUB_REPO}")
    print(f"  Branch: {GITHUB_BRANCH}")
    print(f"  Local:  {TOOL_DIR}")
    print("=" * 55)
    print()

    # Confirm
    print("This will update all files from GitHub.")
    print("Your config files and PROJECTS folder will be kept.")
    print()
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    print()

    # Step 1: Backup
    print("[1] Backing up config files...")
    backups = backup_keep_files()

    # Step 2: Download
    print()
    print("[2] Downloading latest version...")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "repo.zip"

        if not download_zip(ZIP_URL, zip_path):
            print()
            print("[ERROR] Update failed!")
            input("\nPress Enter to exit...")
            return

        # Step 3: Extract
        print()
        print("[3] Extracting files...")
        extracted_dir = extract_zip(zip_path, temp_dir)

        if not extracted_dir:
            print()
            print("[ERROR] Extract failed!")
            input("\nPress Enter to exit...")
            return

        print(f"  Extracted to: {extracted_dir.name}")

        # Step 4: Copy files
        print()
        print("[4] Updating files...")
        copied, skipped = copy_new_files(extracted_dir, TOOL_DIR)
        print(f"  Copied: {copied} files")
        print(f"  Skipped: {skipped} files (kept)")

    # Step 5: Restore backups
    if backups:
        print()
        print("[5] Restoring config files...")
        restore_keep_files(backups)

    # Done
    print()
    print("=" * 55)
    print("  UPDATE COMPLETE!")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
