#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Tool - Thumbnail Generator (Dispatcher)

Workflow:
1. Nhận code (e.g., KA1-0001)
2. Đọc template từ Google Sheets (cột AI) → e.g., KA1-T5
3. Chạy thumb/THUMB_KA1-T5.py với code làm argument
4. Script THUMB tự xử lý và xuất ra thumb/thumbnails/{code}.jpg
5. Copy kết quả về VISUAL/{code}/{code}.jpg

Usage:
    python run_thumb.py KA1-0001    (xử lý 1 mã)
    python run_thumb.py             (scan và xử lý tất cả)
"""

import sys
import os
import shutil
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

TOOL_DIR = Path(__file__).parent
VISUAL_DIR = Path("D:/AUTO/VISUAL")
THUMB_DIR = TOOL_DIR / "thumb"
THUMB_OUTPUT_DIR = THUMB_DIR / "thumbnails"
CONFIG_FILE = TOOL_DIR / "config" / "config.json"

# Google Sheets columns (0-indexed)
COL_CODE = 0        # A: video code
COL_TEMPLATE = 34   # AI: template code (e.g., KA1-T5)

# Source image filename (expandable to multiple later)
THUMB_SOURCE_FILENAME = "thumb_004.png"

SCAN_INTERVAL = 30  # seconds between scans

# Ensure output dir exists
THUMB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subprocess flags for Windows
SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW

# ============================================================================
# LOGGING
# ============================================================================

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


# ============================================================================
# GOOGLE SHEETS
# ============================================================================

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    thumb_config = THUMB_DIR / "config.json"
    if thumb_config.exists():
        with open(thumb_config, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_sheets_data(config):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        log("gspread not installed. Run: pip install gspread google-auth", "ERROR")
        return {}

    cred_path = TOOL_DIR / "config" / config.get("CREDENTIAL_PATH", "creds.json")
    if not cred_path.exists():
        log(f"Credentials not found: {cred_path}", "ERROR")
        return {}

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_file(str(cred_path), scopes=scopes)
        gc = gspread.authorize(creds)
        ws = gc.open(config.get("SPREADSHEET_NAME", "KA")).worksheet(config.get("SHEET_NAME", "INPUT"))
        rows = ws.get_all_values()[1:]  # Skip header

        records = {}
        for r in rows:
            if len(r) <= COL_TEMPLATE:
                continue
            code = r[COL_CODE].strip() if r[COL_CODE] else ""
            if not code:
                continue
            records[code] = {
                "code": code,
                "template": r[COL_TEMPLATE].strip() if len(r) > COL_TEMPLATE else "",
            }

        log(f"Loaded {len(records)} records from Google Sheets")
        return records
    except Exception as e:
        log(f"Failed to load Google Sheets: {e}", "ERROR")
        return {}


# ============================================================================
# TEMPLATE SCRIPT
# ============================================================================

def find_thumb_script(template_code):
    """Find THUMB_{template}.py in thumb/ folder."""
    if not template_code:
        return None

    template_code = template_code.strip().upper()

    # Exact match: THUMB_KA1-T5.py
    script_path = THUMB_DIR / f"THUMB_{template_code}.py"
    if script_path.exists():
        return script_path

    # Case-insensitive search (new THUMB_*.py only)
    for f in THUMB_DIR.glob("THUMB_*.py"):
        if template_code.lower() in f.name.lower():
            return f

    return None


def run_thumb_script(script_path, code, timeout=300):
    """Run THUMB script with code as argument."""
    if not script_path or not script_path.exists():
        return False

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, str(script_path), code],
            cwd=str(THUMB_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            creationflags=SUBPROCESS_FLAGS if sys.platform == "win32" else 0
        )

        if result.returncode == 0:
            return True
        else:
            log(f"Script failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}", "WARN")
            return False

    except subprocess.TimeoutExpired:
        log(f"Script timed out: {script_path.name}", "WARN")
        return False
    except Exception as e:
        log(f"Script error: {e}", "WARN")
        return False


# ============================================================================
# PROJECT DETECTION
# ============================================================================

def get_source_image(project_dir):
    """Find source thumbnail image in VISUAL/{code}/thumbnail/ folder."""
    thumbnail_folder = project_dir / "thumbnail"
    if not thumbnail_folder.exists():
        return None

    # Try main filename first
    main = thumbnail_folder / THUMB_SOURCE_FILENAME
    if main.exists():
        return main

    # Fallback: any image in the folder
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for f in thumbnail_folder.glob(f"*{ext}"):
            return f

    return None


def get_pending_projects(specific_code=None):
    """Get projects that have thumbnail source but no output yet."""
    pending = []

    if not VISUAL_DIR.exists():
        return pending

    for project_dir in VISUAL_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        code = project_dir.name
        if specific_code and code != specific_code:
            continue

        source_img = get_source_image(project_dir)
        if not source_img:
            continue

        # Skip if already done
        final_thumb = project_dir / f"{code}.jpg"
        if final_thumb.exists():
            continue

        pending.append({
            "code": code,
            "project_dir": project_dir,
            "source_img": source_img,
        })

    return pending


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_project(project_info, records):
    """Process single project: find template → run THUMB script → copy output."""
    code = project_info["code"]
    project_dir = project_info["project_dir"]

    print(f"\n[THUMB] {code}")

    # Get template from Google Sheets
    record = records.get(code, {})
    template = record.get("template", "")
    if not template:
        log(f"  No template code in Google Sheets for {code}", "WARN")
        return False

    log(f"  Template: {template}")

    # Find THUMB script
    thumb_script = find_thumb_script(template)
    if not thumb_script:
        log(f"  No THUMB script found for template: {template}", "WARN")
        return False

    log(f"  Running: {thumb_script.name} {code}")

    # Run THUMB script (passes code as argument)
    success = run_thumb_script(thumb_script, code)
    if not success:
        log(f"  THUMB script failed", "WARN")
        return False

    # Check output
    thumb_output = THUMB_OUTPUT_DIR / f"{code}.jpg"
    if not thumb_output.exists():
        log(f"  Output not found: {thumb_output}", "WARN")
        return False

    log(f"  [OK] Thumbnail generated: {thumb_output.name}")

    # Copy to VISUAL/{code}/{code}.jpg (for run_edit.py to pick up)
    if project_dir.exists():
        final_thumb = project_dir / f"{code}.jpg"
        shutil.copy2(thumb_output, final_thumb)
        log(f"  [OK] Copied to VISUAL: {final_thumb.name}")
    else:
        log(f"  [WARN] VISUAL folder gone, skip copy: {project_dir}", "WARN")

    return True


def run_single_project(code):
    print(f"\n{'='*60}")
    print(f"  VE3 TOOL - THUMBNAIL GENERATOR")
    print(f"  Processing: {code}")
    print(f"{'='*60}")

    config = load_config()
    records = get_sheets_data(config)

    pending = get_pending_projects(code)
    if not pending:
        log(f"  Project {code} not found, already done, or no thumbnail/ folder")
        return False

    return process_project(pending[0], records)


def run_scan_loop():
    print(f"\n{'='*60}")
    print(f"  VE3 TOOL - THUMBNAIL GENERATOR (Scan Mode)")
    print(f"  VISUAL:   {VISUAL_DIR}")
    print(f"  Scripts:  {THUMB_DIR}/THUMB_*.py")
    print(f"  Output:   thumb/thumbnails/")
    print(f"  Scan interval: {SCAN_INTERVAL}s")
    print(f"{'='*60}")

    config = load_config()
    records = get_sheets_data(config)

    while True:
        pending = get_pending_projects()

        if not pending:
            log(f"No pending projects. Waiting {SCAN_INTERVAL}s...")
            time.sleep(SCAN_INTERVAL)
            continue

        print(f"\n  Found {len(pending)} project(s):")
        for p in pending:
            print(f"    - {p['code']}")

        success_count = sum(process_project(p, records) for p in pending)
        print(f"\n  Completed: {success_count}/{len(pending)}")
        log("Batch done. Rescanning...")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="VE3 Thumbnail Generator")
    parser.add_argument("code", nargs="?", help="Process specific project code")
    args = parser.parse_args()

    if args.code:
        run_single_project(args.code)
    else:
        run_scan_loop()


if __name__ == "__main__":
    main()
