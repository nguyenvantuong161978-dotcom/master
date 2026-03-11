#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update từ GitHub - Không cần cài Git.
Tải ZIP qua GitHub API (hỗ trợ private repo) và giải nén.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

GITHUB_REPO = "nguyenvantuong161978-dotcom/master"
GITHUB_BRANCH = "main"
ZIP_URL = f"https://api.github.com/repos/{GITHUB_REPO}/zipball/{GITHUB_BRANCH}"

TOOL_DIR = Path(__file__).parent
TOKEN_FILE = TOOL_DIR / "github_token.txt"

# Files/folders không ghi đè khi update
SKIP_PATTERNS = [
    "github_token.txt",
    "progress.json",
    "queue_tracker.json",
    "timing_log.json",
    "run_log.txt",
    "thumb/creds.json",
    "thumb/thumbnails/",
    "thumb/assets/",
    "_designer_uploads/",
]


def get_token():
    """Đọc GitHub token từ file."""
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    return None


def should_skip(rel_path):
    """Kiểm tra file có nên skip không."""
    rel_str = str(rel_path).replace("\\", "/")
    for pattern in SKIP_PATTERNS:
        if pattern.endswith("/"):
            if rel_str.startswith(pattern):
                return True
        else:
            if rel_str == pattern:
                return True
    return False


def download_and_update():
    """Download ZIP từ GitHub API và cập nhật files."""
    token = get_token()
    if not token:
        print("  [ERROR] Chưa có GitHub token!")
        print(f"  Tạo file: {TOKEN_FILE}")
        print("  Nội dung: GitHub Personal Access Token (classic)")
        print("  Tạo tại: https://github.com/settings/tokens")
        print("  Quyền cần: repo (Full control of private repositories)")
        return False

    print(f"  Downloading từ GitHub...")

    # Tạo request với token
    req = urllib.request.Request(ZIP_URL)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")

    import tempfile
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "repo.zip")

    try:
        # Download
        with urllib.request.urlopen(req) as response:
            with open(zip_path, 'wb') as f:
                f.write(response.read())

        size_mb = os.path.getsize(zip_path) / 1024 / 1024
        print(f"  Downloaded: {size_mb:.2f} MB")

        # Extract
        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        # Tìm thư mục giải nén (GitHub tạo folder repo-branch-hash)
        extracted = [f for f in Path(temp_dir).iterdir()
                     if f.is_dir() and f.name != "__MACOSX"]
        if not extracted:
            print("  [ERROR] Không tìm thấy thư mục sau giải nén!")
            return False

        src_dir = extracted[0]
        print(f"  Source: {src_dir.name}")

        # Copy files
        copied = 0
        skipped = 0
        for item in src_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(src_dir)

                if should_skip(rel_path):
                    skipped += 1
                    continue

                dst = TOOL_DIR / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy2(item, dst)
                    copied += 1
                except PermissionError:
                    print(f"  [WARN] Không ghi được: {rel_path} (đang dùng?)")
                except Exception as e:
                    print(f"  [WARN] {rel_path}: {e}")

        print(f"  Copied: {copied} files")
        if skipped:
            print(f"  Skipped: {skipped} files (giữ nguyên)")

        return True

    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("  [ERROR] Token không hợp lệ hoặc hết hạn!")
            print("  Tạo token mới tại: https://github.com/settings/tokens")
        elif e.code == 404:
            print("  [ERROR] Không tìm thấy repo! Kiểm tra tên repo.")
        else:
            print(f"  [ERROR] HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False
    finally:
        # Cleanup temp
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def main():
    print("=" * 55)
    print("  UPDATE FROM GITHUB")
    print("=" * 55)
    print(f"  Local: {TOOL_DIR}")
    print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print()

    success = download_and_update()

    print()
    if success:
        print("=" * 55)
        print("  UPDATE THANH CONG!")
        print("=" * 55)
    else:
        print("  UPDATE THAT BAI!")

    input("\nNhan Enter de dong...")


if __name__ == "__main__":
    main()
