#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Tool - Thumbnail & NV Generator (Dispatcher)

Dispatcher script that uses existing template files:
- 4.NV_{template}.py  -> Creates NV with background removal + name badge
- 7.THUMB_{template}.py -> Creates thumbnail with template-specific design

Workflow:
1. Scan VISUAL for projects with thumb/ subfolder
2. Read Google Sheets to get template code (column AI)
3. Copy source image to thumb/pic/{code}.png
4. Run the matching template scripts
5. Copy results back to VISUAL/{code}/

Usage:
    python run_thumb.py                    (scan and process all)
    python run_thumb.py KA5-0183           (process single project)
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
PIC_DIR = THUMB_DIR / "pic"
NV_OUTPUT_DIR = THUMB_DIR / "nv"
THUMB_OUTPUT_DIR = THUMB_DIR / "thumbnails"
CONFIG_FILE = TOOL_DIR / "config" / "config.json"

# Google Sheets columns (0-indexed)
COL_CODE = 0        # A: video code
COL_TEMPLATE = 34   # AI: template code (e.g., KA1-T5)
COL_NAME = 40       # AO: character name

# Ensure directories exist
PIC_DIR.mkdir(parents=True, exist_ok=True)
NV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subprocess flags for Windows
SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW

# ============================================================================
# LOGGING
# ============================================================================

def log(msg, level="INFO"):
    """Print log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


# ============================================================================
# GOOGLE SHEETS
# ============================================================================

def load_config():
    """Load config from config.json."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Try thumb/config.json
    thumb_config = THUMB_DIR / "config.json"
    if thumb_config.exists():
        with open(thumb_config, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_sheets_data(config):
    """Get all records from Google Sheets INPUT."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        log("gspread not installed. Run: pip install gspread google-auth", "ERROR")
        return {}

    cred_path = config.get("CREDENTIAL_PATH", "creds.json")
    if not Path(cred_path).exists():
        cred_path = TOOL_DIR / "config" / Path(cred_path).name
    if not Path(cred_path).exists():
        cred_path = THUMB_DIR / Path(cred_path).name

    if not Path(cred_path).exists():
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

        # Parse rows into records keyed by code
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
                "name": r[COL_NAME].strip() if len(r) > COL_NAME else code,
            }

        log(f"Loaded {len(records)} records from Google Sheets")
        return records
    except Exception as e:
        log(f"Failed to load Google Sheets: {e}", "ERROR")
        return {}


# ============================================================================
# TEMPLATE SCRIPTS
# ============================================================================

def find_template_script(script_type, template_code):
    """
    Find template script file.
    script_type: "NV" or "THUMB"
    template_code: e.g., "KA1-T5", "KA2-T2"

    Returns path to script like 4.NV_KA1-T5.py or 7.THUMB_KA2-T2.py
    """
    if not template_code:
        return None

    # Normalize template code
    template_code = template_code.strip().upper()

    # Try exact match first
    if script_type == "NV":
        pattern = f"4.NV_{template_code}.py"
    else:
        pattern = f"7.THUMB_{template_code}.py"

    script_path = THUMB_DIR / pattern
    if script_path.exists():
        return script_path

    # Try case-insensitive search
    for f in THUMB_DIR.glob("*.py"):
        if script_type == "NV" and f.name.startswith("4.NV_"):
            if template_code.lower() in f.name.lower():
                return f
        elif script_type == "THUMB" and f.name.startswith("7.THUMB_"):
            if template_code.lower() in f.name.lower():
                return f

    return None


def run_template_script(script_path, timeout=300):
    """Run a template script."""
    if not script_path or not script_path.exists():
        return False

    try:
        # Set PYTHONIOENCODING to handle Unicode on Windows
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(THUMB_DIR),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
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
# MAIN WORKFLOW
# ============================================================================

def find_source_image(thumb_folder, nv_folder=None):
    """Find source image in thumb folder (nv1.* or first image).
    Falls back to nv folder if thumb folder doesn't exist or has no images.
    """
    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}

    # Check thumb folder first
    if thumb_folder and thumb_folder.exists():
        # Try nv1.* first
        for ext in valid_ext:
            path = thumb_folder / f"nv1{ext}"
            if path.exists():
                return path

        # Fallback to first image in thumb folder
        for f in thumb_folder.iterdir():
            if f.is_file() and f.suffix.lower() in valid_ext:
                return f

    # Fallback to nv folder if provided
    if nv_folder and nv_folder.exists():
        for ext in valid_ext:
            path = nv_folder / f"nv1{ext}"
            if path.exists():
                return path
        # Try any image in nv folder
        for f in nv_folder.iterdir():
            if f.is_file() and f.suffix.lower() in valid_ext:
                return f

    return None


def get_pending_projects(specific_code=None):
    """Get projects that need thumbnail/NV processing."""
    pending = []

    if not VISUAL_DIR.exists():
        return pending

    for project_dir in VISUAL_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        code = project_dir.name

        # Filter for specific code if provided
        if specific_code and code != specific_code:
            continue

        thumb_folder = project_dir / "thumb"
        nv_folder = project_dir / "nv"

        # Find source image (from thumb/ or fallback to nv/)
        source_img = find_source_image(thumb_folder, nv_folder)
        if not source_img:
            continue

        # Check if already done (output exists in VISUAL folder)
        final_thumb = project_dir / f"{code}.jpg"
        if final_thumb.exists():
            continue

        pending.append({
            "code": code,
            "project_dir": project_dir,
            "thumb_folder": thumb_folder,
            "source_img": source_img,
        })

    return pending


def process_project(project_info, records):
    """Process single project for NV and Thumbnail."""
    code = project_info["code"]
    source_img = project_info["source_img"]
    project_dir = project_info["project_dir"]

    print(f"\n[THUMB] {code}")

    # Get record from Google Sheets
    record = records.get(code, {})
    template = record.get("template", "")
    name = record.get("name", code)

    if not template:
        log(f"  No template code in Google Sheets for {code}", "WARN")
        return False

    print(f"  Template: {template}")
    print(f"  Name: {name}")

    # Step 1: Copy source image to thumb/pic/{code}.png
    dest_img = PIC_DIR / f"{code}.png"
    if not dest_img.exists():
        print(f"  Copying source to pic/{code}.png")
        # Convert to PNG if needed
        try:
            from PIL import Image
            img = Image.open(source_img)
            img.save(dest_img, "PNG")
        except:
            shutil.copy2(source_img, dest_img)

    # Step 2: Find and run NV template script
    nv_script = find_template_script("NV", template)
    nv_output = NV_OUTPUT_DIR / f"{code}.png"

    if nv_script:
        print(f"  Running NV script: {nv_script.name}")
        if not nv_output.exists():
            run_template_script(nv_script)

        if nv_output.exists():
            print(f"  [OK] NV generated: {nv_output.name}")
        else:
            print(f"  [WARN] NV not generated")
    else:
        print(f"  [WARN] No NV script found for template: {template}")

    # Step 3: Find and run THUMB template script
    thumb_script = find_template_script("THUMB", template)
    thumb_output = THUMB_OUTPUT_DIR / f"{code}.jpg"

    if thumb_script:
        print(f"  Running THUMB script: {thumb_script.name}")
        if not thumb_output.exists():
            run_template_script(thumb_script)

        if thumb_output.exists():
            print(f"  [OK] Thumbnail generated: {thumb_output.name}")
        else:
            print(f"  [WARN] Thumbnail not generated")
    else:
        print(f"  [WARN] No THUMB script found for template: {template}")

    # Step 4: Copy outputs to VISUAL/{code}/
    final_thumb = project_dir / f"{code}.jpg"
    final_nv = project_dir / f"{code}_nv.png"

    if thumb_output.exists() and not final_thumb.exists():
        shutil.copy2(thumb_output, final_thumb)
        print(f"  [OK] Copied to VISUAL: {final_thumb.name}")

    if nv_output.exists() and not final_nv.exists():
        shutil.copy2(nv_output, final_nv)
        print(f"  [OK] Copied to VISUAL: {final_nv.name}")

    return final_thumb.exists() or final_nv.exists()


SCAN_INTERVAL = 30  # seconds between scans


def run_single_project(code):
    """Process a single project by code."""
    print(f"\n{'='*60}")
    print(f"  VE3 TOOL - THUMBNAIL & NV GENERATOR")
    print(f"  Processing: {code}")
    print(f"{'='*60}")

    config = load_config()
    records = get_sheets_data(config)

    pending = get_pending_projects(code)
    if not pending:
        print(f"  Project {code} not found or already processed.")
        return

    for project_info in pending:
        process_project(project_info, records)


def run_scan_loop():
    """Run continuous scan loop for batch processing."""
    print(f"\n{'='*60}")
    print(f"  VE3 TOOL - THUMBNAIL & NV GENERATOR (Dispatcher)")
    print(f"{'='*60}")
    print(f"  VISUAL:   {VISUAL_DIR}")
    print(f"  Templates: {THUMB_DIR}")
    print(f"  Output:   VISUAL/{{code}}/")
    print(f"  Scan interval: {SCAN_INTERVAL}s")
    print(f"{'='*60}")

    # Load config and Google Sheets data once
    config = load_config()
    records = get_sheets_data(config)

    while True:
        # Get pending projects
        pending = get_pending_projects()

        if not pending:
            log(f"No pending projects. Waiting {SCAN_INTERVAL}s...")
            time.sleep(SCAN_INTERVAL)
            continue

        print(f"\n  Found {len(pending)} project(s) to process:")
        for p in pending:
            print(f"    - {p['code']}")

        # Process each
        success_count = 0
        for project_info in pending:
            if process_project(project_info, records):
                success_count += 1

        print(f"\n  Completed: {success_count}/{len(pending)}")

        # Short delay before next scan (immediate rescan after batch)
        log(f"Batch done. Rescanning...")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="VE3 Thumbnail & NV Generator")
    parser.add_argument("code", nargs="?", help="Process specific project code")
    args = parser.parse_args()

    if args.code:
        run_single_project(args.code)
    else:
        run_scan_loop()


if __name__ == "__main__":
    main()
