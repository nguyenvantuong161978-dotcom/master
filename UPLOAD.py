#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload to GitHub without Git installed.
Uses GitHub API with Personal Access Token.
"""

import os
import sys
import base64
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG - Edit these values
# ============================================================================

GITHUB_REPO = "nguyenvantuong161978-dotcom/master"
GITHUB_BRANCH = "main"

# Token file path (create this file with your GitHub token)
TOKEN_FILE = Path(__file__).parent / "config" / "github_token.txt"

# Files to upload (relative to this script)
FILES_TO_UPLOAD = [
    "run_srt.py",
    "run_edit.py",
    "GUI.pyw",
    "RUN_MASTER.bat",
    "RUN_GUI.bat",
    "UPLOAD.bat",
    "UPLOAD.py",
    "UPDATE.py",
    "UPDATE.bat",
    ".gitignore",
    "config/settings.yaml",
]


# ============================================================================
# GITHUB API
# ============================================================================

def get_token():
    """Get GitHub token from file."""
    if not TOKEN_FILE.exists():
        print(f"[ERROR] Token file not found: {TOKEN_FILE}")
        print()
        print("Please create the file with your GitHub Personal Access Token:")
        print(f"  1. Go to: https://github.com/settings/tokens")
        print(f"  2. Generate new token (classic) with 'repo' scope")
        print(f"  3. Save token to: {TOKEN_FILE}")
        return None

    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        print("[ERROR] Token file is empty!")
        return None

    return token


def github_api(method, endpoint, token, data=None):
    """Make GitHub API request."""
    url = f"https://api.github.com{endpoint}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "VE3-Tool-Uploader",
    }

    if data:
        data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"[API ERROR] {e.code}: {error_body[:200]}")
        return None


def get_file_sha(token, path):
    """Get SHA of existing file (needed for update)."""
    result = github_api("GET", f"/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}", token)
    if result and "sha" in result:
        return result["sha"]
    return None


def upload_file(token, local_path, remote_path):
    """Upload a single file to GitHub."""
    local_file = Path(__file__).parent / local_path

    if not local_file.exists():
        print(f"  [SKIP] {local_path} - file not found")
        return False

    # Read and encode file
    content = local_file.read_bytes()
    content_b64 = base64.b64encode(content).decode("utf-8")

    # Check if file exists (get SHA for update)
    sha = get_file_sha(token, remote_path)

    # Prepare request
    data = {
        "message": f"Update {remote_path} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": content_b64,
        "branch": GITHUB_BRANCH,
    }

    if sha:
        data["sha"] = sha
        action = "UPDATE"
    else:
        action = "CREATE"

    # Upload
    result = github_api("PUT", f"/repos/{GITHUB_REPO}/contents/{remote_path}", token, data)

    if result and "content" in result:
        print(f"  [OK] {local_path} -> {remote_path} ({action})")
        return True
    else:
        print(f"  [FAIL] {local_path}")
        return False


def ensure_branch_exists(token):
    """Create main branch if not exists."""
    # Check if branch exists
    result = github_api("GET", f"/repos/{GITHUB_REPO}/branches/{GITHUB_BRANCH}", token)
    if result:
        return True

    # Get default branch SHA
    repo_info = github_api("GET", f"/repos/{GITHUB_REPO}", token)
    if not repo_info:
        print("[ERROR] Cannot get repo info")
        return False

    default_branch = repo_info.get("default_branch", "main")

    # Get SHA of default branch
    ref_info = github_api("GET", f"/repos/{GITHUB_REPO}/git/refs/heads/{default_branch}", token)
    if not ref_info:
        # Repo might be empty, create initial commit
        print("[INFO] Creating initial commit...")

        # Create a blob
        blob = github_api("POST", f"/repos/{GITHUB_REPO}/git/blobs", token, {
            "content": "# VE3 Tool Master\n\nVideo editing automation tool.",
            "encoding": "utf-8"
        })

        if not blob:
            return False

        # Create tree
        tree = github_api("POST", f"/repos/{GITHUB_REPO}/git/trees", token, {
            "tree": [{"path": "README.md", "mode": "100644", "type": "blob", "sha": blob["sha"]}]
        })

        if not tree:
            return False

        # Create commit
        commit = github_api("POST", f"/repos/{GITHUB_REPO}/git/commits", token, {
            "message": "Initial commit",
            "tree": tree["sha"]
        })

        if not commit:
            return False

        # Create ref
        github_api("POST", f"/repos/{GITHUB_REPO}/git/refs", token, {
            "ref": f"refs/heads/{GITHUB_BRANCH}",
            "sha": commit["sha"]
        })

        return True

    return True


def main():
    print("=" * 50)
    print("  UPLOAD TO GITHUB")
    print("=" * 50)
    print(f"  Repo:   {GITHUB_REPO}")
    print(f"  Branch: {GITHUB_BRANCH}")
    print("=" * 50)
    print()

    # Get token
    token = get_token()
    if not token:
        input("\nPress Enter to exit...")
        return

    print("[1] Checking repository...")
    if not ensure_branch_exists(token):
        print("[ERROR] Cannot access repository!")
        input("\nPress Enter to exit...")
        return

    print("[2] Uploading files...")
    success = 0
    failed = 0

    for file_path in FILES_TO_UPLOAD:
        if upload_file(token, file_path, file_path):
            success += 1
        else:
            failed += 1

    print()
    print("=" * 50)
    print(f"  DONE: {success} uploaded, {failed} failed")
    print(f"  URL: https://github.com/{GITHUB_REPO}")
    print("=" * 50)

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
