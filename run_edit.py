#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Tool - MASTER: Edit Mode (Compose MP4)
Quét VISUAL folder và ghép video từ ảnh + voice + SRT.

Usage:
    python run_edit.py                     (quét và xử lý tự động)
    python run_edit.py AR47-0028           (chạy 1 project cụ thể)
    python run_edit.py --parallel 3        (chạy 3 project song song)
    python run_edit.py --scan-only         (chỉ quét, không xử lý)
"""

import sys
import os
import time
import shutil
import json
import re
import subprocess
import argparse
import random
import tempfile
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# Import Ken Burns CV2 module
try:
    from modules.ken_burns_cv2 import KenBurnsCv2, QUALITY_PRESETS
    KEN_BURNS_CV2_AVAILABLE = True
except ImportError:
    KEN_BURNS_CV2_AVAILABLE = False
    print("[WARN] ken_burns_cv2 module not found, using FFmpeg fallback")

# ============================================================================
# CONFIG
# ============================================================================

TOOL_DIR = Path(__file__).parent
VISUAL_DIR = Path(r"D:\AUTO\VISUAL")
DONE_DIR = Path(r"D:\AUTO\done")
THUMB_DIR = Path(r"D:\AUTO\thumbnails")
VOICE_DIR = Path(r"D:\AUTO\voice")  # Voice source folder
PROJECTS_DIR = TOOL_DIR / "PROJECTS"  # SRT projects folder
CONFIG_FILE = TOOL_DIR / "config" / "config.json"
PROGRESS_FILE = TOOL_DIR / "progress.json"

SCAN_INTERVAL = 30  # Scan every 30 seconds for new projects
DEFAULT_PARALLEL = 4
CLIP_WORKERS = 3  # Number of parallel workers for clip creation (3 workers ~ 90% CPU)

# Google Sheet config
SOURCE_SHEET_NAME = "NGUON"
SOURCE_COL_CODE = 7
SOURCE_COL_STATUS = 13
STATUS_VALUE = "EDIT XONG"

MAX_RETRIES = 7  # Increased for Google Sheets reliability
RETRY_BASE_DELAY = 3  # Start with 3s delay

# Hide console window for subprocess on Windows
if sys.platform == "win32":
    SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW
else:
    SUBPROCESS_FLAGS = 0


# Progress tracking for GUI
_current_progress = {
    "code": "",
    "step": "",
    "percent": 0,
    "clip_current": 0,
    "clip_total": 0,
    "status": "idle"
}


def update_progress(code: str = None, step: str = None, percent: int = None,
                   clip_current: int = None, clip_total: int = None, status: str = None):
    """Update progress and write to file for GUI to read."""
    if code is not None:
        _current_progress["code"] = code
    if step is not None:
        _current_progress["step"] = step
    if percent is not None:
        _current_progress["percent"] = percent
    if clip_current is not None:
        _current_progress["clip_current"] = clip_current
    if clip_total is not None:
        _current_progress["clip_total"] = clip_total
    if status is not None:
        _current_progress["status"] = status

    _current_progress["updated"] = time.strftime("%H:%M:%S")

    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(_current_progress, f)
    except:
        pass


def log(msg: str, level: str = "INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


# ============================================================================
# SUBTITLE TEMPLATE SYSTEM
# ============================================================================

SUBTITLE_TEMPLATES_FILE = TOOL_DIR / "subtitle_templates.json"

# Default template (used when no channel-specific template exists)
DEFAULT_SUBTITLE_TEMPLATE = {
    "font": "Bebas Neue",
    "size": 28,
    "color": "&H00FFFFFF",      # White (ABGR format)
    "outline": "&H00000000",    # Black outline
    "outline_size": 2,
    "margin_v": 25,
    "alignment": 2,             # 2 = bottom center
    # Video settings (per channel)
    "output_resolution": "4k",
    "compose_mode": "quality",
    "ken_burns_intensity": "subtle",
    "video_transition": "random",
    # NV overlay settings
    "nv_overlay_enabled": True,
    "nv_overlay_position": "left",
    "nv_overlay_v_position": "middle",
    "nv_overlay_scale": 0.50,
    "nv_crop_ratio": 0.5,  # Crop right portion (0.5 = right half, 1.0 = full image)
}

# Available fonts in fonts/ folder
AVAILABLE_FONTS = [
    "Bebas Neue",
    "Inter Bold",
    "Noto Serif",
    "Anton",
    "League Spartan ExtraBold",
    "Montserrat",
    "Nunito",
    "Roboto Condensed",
    "UTM Avo Bold",
    "Zuume SemiBold"
]

# Alignment options: 1=left, 2=center, 3=right (bottom row)
# 4=left, 5=center, 6=right (middle row)
# 7=left, 8=center, 9=right (top row)
ALIGNMENT_OPTIONS = {
    "bottom_left": 1,
    "bottom_center": 2,
    "bottom_right": 3,
    "middle_left": 4,
    "middle_center": 5,
    "middle_right": 6,
    "top_left": 7,
    "top_center": 8,
    "top_right": 9
}


def load_subtitle_templates() -> Dict:
    """Load subtitle templates from JSON file."""
    if SUBTITLE_TEMPLATES_FILE.exists():
        try:
            with open(SUBTITLE_TEMPLATES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_subtitle_templates(templates: Dict):
    """Save subtitle templates to JSON file."""
    try:
        with open(SUBTITLE_TEMPLATES_FILE, "w", encoding="utf-8") as f:
            json.dump(templates, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"Error saving templates: {e}", "ERROR")


def get_subtitle_template(code: str) -> Dict:
    """Get subtitle template for a channel based on code prefix (e.g., KA1, KA2)."""
    templates = load_subtitle_templates()

    # Extract channel prefix (e.g., "KA1" from "KA1-0023")
    channel = code.split("-")[0] if "-" in code else code

    # Look for exact channel match
    if channel in templates:
        template = DEFAULT_SUBTITLE_TEMPLATE.copy()
        template.update(templates[channel])
        return template

    # Look for base channel (e.g., "KA" from "KA1")
    base_channel = ''.join(c for c in channel if not c.isdigit())
    if base_channel in templates:
        template = DEFAULT_SUBTITLE_TEMPLATE.copy()
        template.update(templates[base_channel])
        return template

    return DEFAULT_SUBTITLE_TEMPLATE.copy()


def set_subtitle_template(channel: str, template: Dict):
    """Set subtitle template for a channel."""
    templates = load_subtitle_templates()
    templates[channel] = template
    save_subtitle_templates(templates)
    log(f"Saved template for channel: {channel}")


def get_all_templates() -> Dict:
    """Get all saved templates."""
    return load_subtitle_templates()


def normalize_code(code: str) -> str:
    if not code:
        return ""
    s = str(code)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


# ============================================================================
# KEN BURNS EFFECT (inline from modules/ken_burns.py)
# ============================================================================

class KenBurnsEffect(Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    ZOOM_IN_LEFT = "zoom_in_left"
    ZOOM_IN_RIGHT = "zoom_in_right"
    ZOOM_OUT_CENTER = "zoom_out_center"


class KenBurnsIntensity(Enum):
    SUBTLE = "subtle"
    NORMAL = "normal"
    STRONG = "strong"


INTENSITY_SETTINGS = {
    KenBurnsIntensity.SUBTLE: (0.05, 0.03),
    KenBurnsIntensity.NORMAL: (0.12, 0.08),
    KenBurnsIntensity.STRONG: (0.20, 0.15),
}


class KenBurnsGenerator:
    def __init__(self, width: int = 1920, height: int = 1080,
                 intensity: str = "normal", fps: int = 25):
        self.width = width
        self.height = height
        self.fps = fps

        if isinstance(intensity, str):
            intensity = intensity.lower()
            self.intensity = {
                "subtle": KenBurnsIntensity.SUBTLE,
                "normal": KenBurnsIntensity.NORMAL,
                "strong": KenBurnsIntensity.STRONG,
            }.get(intensity, KenBurnsIntensity.NORMAL)
        else:
            self.intensity = intensity

        self.zoom_percent, self.pan_percent = INTENSITY_SETTINGS[self.intensity]

    def get_random_effect(self, exclude_last=None):
        effects = list(KenBurnsEffect)
        if exclude_last and exclude_last in effects:
            effects.remove(exclude_last)
        return random.choice(effects)

    def generate_filter(self, effect, duration: float,
                       fade_duration: float = 0.5, simple_mode: bool = False) -> str:
        w, h = self.width, self.height
        total_frames = int(duration * self.fps)

        zoom_start = 1.0
        zoom_end = 1.0 + self.zoom_percent
        pan_x = int(w * self.pan_percent)
        pan_y = int(h * self.pan_percent)

        if simple_mode:
            zoom_expr, x_expr, y_expr = self._get_linear_expressions(
                effect, zoom_start, zoom_end, pan_x, pan_y, total_frames
            )
        else:
            zoom_expr, x_expr, y_expr = self._get_eased_expressions(
                effect, zoom_start, zoom_end, pan_x, pan_y, total_frames
            )

        zoompan = f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={w}x{h}:fps={self.fps}"
        fade_out_start = max(0, duration - fade_duration)
        fade_filter = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out_start}:d={fade_duration}"

        return f"{zoompan},{fade_filter}"

    def _get_linear_expressions(self, effect, zoom_start, zoom_end, pan_x, pan_y, total_frames):
        progress = f"on/{total_frames}"

        if effect == KenBurnsEffect.ZOOM_IN:
            zoom = f"{zoom_start}+{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)"
            y = f"ih/2-(ih/zoom/2)"
        elif effect == KenBurnsEffect.ZOOM_OUT:
            zoom = f"{zoom_end}-{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)"
            y = f"ih/2-(ih/zoom/2)"
        elif effect == KenBurnsEffect.PAN_LEFT:
            zoom = str(zoom_start)
            x = f"{pan_x}*(1-{progress})"
            y = "0"
        elif effect == KenBurnsEffect.PAN_RIGHT:
            zoom = str(zoom_start)
            x = f"{pan_x}*{progress}"
            y = "0"
        elif effect == KenBurnsEffect.PAN_UP:
            zoom = str(zoom_start)
            x = "0"
            y = f"{pan_y}*(1-{progress})"
        elif effect == KenBurnsEffect.PAN_DOWN:
            zoom = str(zoom_start)
            x = "0"
            y = f"{pan_y}*{progress}"
        elif effect == KenBurnsEffect.ZOOM_IN_LEFT:
            zoom = f"{zoom_start}+{zoom_end - zoom_start}*{progress}"
            x = f"(iw/4)*(1-{progress})"
            y = f"ih/2-(ih/zoom/2)"
        elif effect == KenBurnsEffect.ZOOM_IN_RIGHT:
            zoom = f"{zoom_start}+{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)+{pan_x}*{progress}"
            y = f"ih/2-(ih/zoom/2)"
        else:
            zoom = f"{zoom_end}-{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)"
            y = f"ih/2-(ih/zoom/2)"

        return zoom, x, y

    def _get_eased_expressions(self, effect, zoom_start, zoom_end, pan_x, pan_y, total_frames):
        progress = f"(1-cos(PI*on/{total_frames}))/2"

        if effect == KenBurnsEffect.ZOOM_IN:
            zoom = f"{zoom_start}+{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)"
            y = f"ih/2-(ih/zoom/2)"
        elif effect == KenBurnsEffect.ZOOM_OUT:
            zoom = f"{zoom_end}-{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)"
            y = f"ih/2-(ih/zoom/2)"
        elif effect == KenBurnsEffect.PAN_LEFT:
            zoom = str(zoom_start)
            x = f"{pan_x}*(1-{progress})"
            y = "0"
        elif effect == KenBurnsEffect.PAN_RIGHT:
            zoom = str(zoom_start)
            x = f"{pan_x}*{progress}"
            y = "0"
        elif effect == KenBurnsEffect.PAN_UP:
            zoom = str(zoom_start)
            x = "0"
            y = f"{pan_y}*(1-{progress})"
        elif effect == KenBurnsEffect.PAN_DOWN:
            zoom = str(zoom_start)
            x = "0"
            y = f"{pan_y}*{progress}"
        elif effect == KenBurnsEffect.ZOOM_IN_LEFT:
            zoom = f"{zoom_start}+{zoom_end - zoom_start}*{progress}"
            x = f"(iw/4)*(1-{progress})"
            y = f"ih/2-(ih/zoom/2)"
        elif effect == KenBurnsEffect.ZOOM_IN_RIGHT:
            zoom = f"{zoom_start}+{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)+{pan_x}*{progress}"
            y = f"ih/2-(ih/zoom/2)"
        else:
            zoom = f"{zoom_end}-{zoom_end - zoom_start}*{progress}"
            x = f"iw/2-(iw/zoom/2)"
            y = f"ih/2-(ih/zoom/2)"

        return zoom, x, y


# ============================================================================
# FILL MISSING MEDIA
# ============================================================================

def get_required_scene_ids(excel_path: Path) -> Set[str]:
    """Get all scene IDs required from Excel."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(excel_path)

        scenes_sheet = None
        for sheet_name in wb.sheetnames:
            if 'scene' in sheet_name.lower():
                scenes_sheet = wb[sheet_name]
                break

        if not scenes_sheet:
            return set()

        headers = [cell.value for cell in scenes_sheet[1]]
        id_col = None
        for i, h in enumerate(headers):
            if h and str(h).lower().strip() in ['scene_id', 'id']:
                id_col = i
                break

        if id_col is None:
            return set()

        scene_ids = set()
        for row in scenes_sheet.iter_rows(min_row=2, values_only=True):
            if row[id_col] is not None:
                try:
                    scene_id = str(int(float(str(row[id_col]).strip())))
                    scene_ids.add(scene_id)
                except ValueError:
                    continue
        return scene_ids
    except Exception as e:
        log(f"Error reading Excel: {e}", "WARN")
        return set()


def get_existing_media(img_dir: Path) -> Dict[str, Path]:
    """Get existing media files mapped by scene ID."""
    media = {}
    if not img_dir.exists():
        return media

    for ext in [".mp4", ".png", ".jpg", ".jpeg", ".webp"]:
        for f in img_dir.glob(f"*{ext}"):
            if f.stem.startswith('nv') or f.stem.startswith('loc'):
                continue
            scene_id = f.stem
            if scene_id not in media:
                media[scene_id] = f
    return media


def fill_missing_media(project_dir: Path, excel_path: Path) -> Tuple[int, int]:
    """Fill missing media by copying random existing media files."""
    # Support both VM structure (img\ subfolder) and direct structure
    img_dir = project_dir / "img"
    if not img_dir.exists():
        img_dir = project_dir  # Fallback to root folder

    # Check if there are any media files
    has_media = any(img_dir.glob("*.mp4")) or any(img_dir.glob("*.png")) or any(img_dir.glob("*.jpg"))
    if not has_media:
        log(f"  [FILL] No media found in project", "WARN")
        return 0, 0

    required_ids = get_required_scene_ids(excel_path)
    if not required_ids:
        log(f"  [FILL] No scenes found in Excel", "WARN")
        return 0, 0

    existing_media = get_existing_media(img_dir)
    existing_ids = set(existing_media.keys())
    missing_ids = required_ids - existing_ids

    if not missing_ids:
        log(f"  [FILL] All {len(required_ids)} scenes have media")
        return 0, 0

    if not existing_media:
        log(f"  [FILL] No existing media to copy from!", "ERROR")
        return 0, len(missing_ids)

    log(f"  [FILL] Missing {len(missing_ids)} scenes, filling from {len(existing_media)} existing...")

    existing_files = list(existing_media.values())
    filled_count = 0

    for missing_id in sorted(missing_ids, key=lambda x: int(x) if x.isdigit() else 0):
        source_file = random.choice(existing_files)
        dest_file = img_dir / f"{missing_id}{source_file.suffix}"

        try:
            shutil.copy2(source_file, dest_file)
            log(f"    Copied {source_file.name} -> {dest_file.name}")
            filled_count += 1
        except Exception as e:
            log(f"    Failed to copy {missing_id}: {e}", "WARN")

    log(f"  [FILL] Filled {filled_count}/{len(missing_ids)} missing scenes")
    return filled_count, len(missing_ids) - filled_count


# ============================================================================
# PROJECT DETECTION
# ============================================================================

def get_project_info(project_dir: Path) -> Dict:
    """Get project info from directory."""
    code = project_dir.name

    info = {
        "code": code,
        "path": project_dir,
        "has_srt": False,
        "has_audio": False,
        "has_excel": False,
        "video_count": 0,
        "image_count": 0,
        "media_count": 0,
        "total_scenes": 0,
        "ready_for_edit": False,
        "already_done": False,
    }

    srt_path = project_dir / f"{code}.srt"
    audio_path = project_dir / f"{code}.mp3"
    excel_path = project_dir / f"{code}_prompts.xlsx"

    info["has_srt"] = srt_path.exists()
    info["has_audio"] = audio_path.exists()
    info["has_excel"] = excel_path.exists()
    info["srt_path"] = srt_path if srt_path.exists() else None
    info["audio_path"] = audio_path if audio_path.exists() else None
    info["excel_path"] = excel_path if excel_path.exists() else None

    # Support both VM structure (img\ subfolder) and direct structure
    img_dir = project_dir / "img"
    if not img_dir.exists():
        img_dir = project_dir  # Fallback to root folder

    if img_dir.exists():
        videos = [f for f in img_dir.glob("*.mp4")
                  if not f.stem.startswith('nv') and not f.stem.startswith('loc')]
        # Support multiple image formats
        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            images.extend([f for f in img_dir.glob(ext)
                          if not f.stem.startswith('nv') and not f.stem.startswith('loc')])
        info["video_count"] = len(videos)
        info["image_count"] = len(images)
        info["media_count"] = len(videos) + len(images)

        if excel_path.exists():
            required_ids = get_required_scene_ids(excel_path)
            info["total_scenes"] = len(required_ids)

    done_dir = DONE_DIR / code
    if done_dir.exists():
        mp4_files = list(done_dir.glob("*.mp4"))
        info["already_done"] = len(mp4_files) > 0

    if info["media_count"] > 0 and info["has_audio"] and info["has_excel"]:
        if info["total_scenes"] > 0:
            coverage = info["media_count"] / info["total_scenes"]

            # Fill missing media if coverage >= 10% and < 100%
            if 0.1 <= coverage < 1.0:
                log(f"    - {code}: Coverage {coverage:.0%} < 100%, filling missing...")
                filled, still_missing = fill_missing_media(project_dir, excel_path)

                if filled > 0:
                    videos = [f for f in img_dir.glob("*.mp4")
                              if not f.stem.startswith('nv') and not f.stem.startswith('loc')]
                    images = []
                    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
                        images.extend([f for f in img_dir.glob(ext)
                                      if not f.stem.startswith('nv') and not f.stem.startswith('loc')])
                    info["video_count"] = len(videos)
                    info["image_count"] = len(images)
                    info["media_count"] = len(videos) + len(images)
                    coverage = info["media_count"] / info["total_scenes"]
                    log(f"    - {code}: After fill, coverage is now {coverage:.0%}")

            info["ready_for_edit"] = coverage >= 0.8
        else:
            info["ready_for_edit"] = True

    return info


def scan_visual_projects() -> List[Dict]:
    """Scan VISUAL folder for projects ready to edit."""
    projects = []

    if not VISUAL_DIR.exists():
        log(f"VISUAL folder not found: {VISUAL_DIR}", "WARN")
        return projects

    all_folders = [item for item in VISUAL_DIR.iterdir() if item.is_dir()]
    log(f"  [DEBUG] Found {len(all_folders)} folders in VISUAL")

    for item in all_folders:
        info = get_project_info(item)
        code = info["code"]

        if info["already_done"]:
            log(f"    - {code}: already done")
        elif info["ready_for_edit"]:
            log(f"    - {code}: ready ({info['video_count']}v + {info['image_count']}i / {info['total_scenes']} scenes)")
            projects.append(info)
        else:
            reasons = []
            if info["media_count"] == 0:
                reasons.append("no media")
            if not info["has_audio"]:
                reasons.append("no audio")
            if not info["has_excel"]:
                reasons.append("no excel")
            if info["total_scenes"] > 0 and info["media_count"] > 0:
                coverage = info["media_count"] / info["total_scenes"]
                if coverage < 0.8:
                    reasons.append(f"coverage {coverage:.0%} < 80%")
            log(f"    - {code}: NOT ready ({', '.join(reasons)})")

    return sorted(projects, key=lambda x: x["code"])


# ============================================================================
# VIDEO COMPOSITION
# ============================================================================

def parse_timestamp(timestamp: str) -> float:
    """Parse SRT timestamp to seconds."""
    if not timestamp:
        return 0.0
    timestamp = timestamp.replace(",", ".")
    parts = timestamp.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(timestamp) if timestamp else 0.0


def process_srt_for_video(srt_path: Path, output_path: Path, max_chars: int = 50) -> Path:
    """Process SRT: split long lines."""
    def parse_time(time_str: str) -> float:
        h, m, s = time_str.replace(',', '.').split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

    def split_text(text: str, max_len: int) -> list:
        words = text.split()
        chunks = []
        current = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current else 0)
            if current_len + word_len <= max_len:
                current.append(word)
                current_len += word_len
            else:
                if current:
                    chunks.append(' '.join(current))
                current = [word]
                current_len = len(word)

        if current:
            chunks.append(' '.join(current))

        return chunks if chunks else [text[:max_len]]

    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
        entries = re.findall(pattern, content, re.DOTALL)

        new_entries = []
        new_index = 1

        for idx, start, end, text in entries:
            text = text.strip().replace('\n', ' ').upper()
            start_sec = parse_time(start)
            end_sec = parse_time(end)
            duration = end_sec - start_sec

            if len(text) <= max_chars:
                new_entries.append((new_index, start, end, text))
                new_index += 1
            else:
                chunks = split_text(text, max_chars)
                chunk_duration = duration / len(chunks)

                for i, chunk in enumerate(chunks):
                    chunk_start = start_sec + i * chunk_duration
                    chunk_end = start_sec + (i + 1) * chunk_duration
                    new_entries.append((new_index, format_time(chunk_start), format_time(chunk_end), chunk))
                    new_index += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, start, end, text in new_entries:
                f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

        return output_path

    except Exception as e:
        return srt_path


def compose_video(project_info: Dict, callback=None) -> Tuple[bool, Optional[Path], Optional[str]]:
    """Compose final video."""
    code = project_info["code"]
    project_dir = project_info["path"]
    excel_path = project_info.get("excel_path")

    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(f"[{code}] {msg}", level)

    update_progress(code=code, step="Starting", percent=0, status="composing")
    plog("Starting video composition...")

    if not excel_path or not excel_path.exists():
        return False, None, "Excel file not found"

    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        if result.returncode != 0:
            return False, None, "FFmpeg not working"
    except FileNotFoundError:
        return False, None, "FFmpeg not installed"

    # Find voice file
    voice_files = list(project_dir.glob("*.mp3")) + list(project_dir.glob("*.wav"))
    if not voice_files:
        return False, None, "No voice file found"
    voice_path = voice_files[0]

    # Find SRT file
    srt_files = list(project_dir.glob("srt/*.srt")) + list(project_dir.glob("*.srt"))
    srt_path = srt_files[0] if srt_files else None

    if srt_path:
        processed_srt = project_dir / f"{code}.srt"
        srt_path = process_srt_for_video(srt_path, processed_srt, max_chars=50)

    output_path = project_dir / f"{code}.mp4"

    # Support both VM structure (img\ subfolder) and direct structure
    img_dir = project_dir / "img"
    if not img_dir.exists():
        # Fallback to project_dir itself if no img\ subfolder
        img_dir = project_dir
        plog(f"  Using root folder for media (no img\\ subfolder)")

    plog(f"  Voice: {voice_path.name}")
    plog(f"  SRT: {srt_path.name if srt_path else 'None'}")
    plog(f"  Excel: {excel_path.name}")

    # Check if Excel file is stable (not being written/downloaded)
    def is_file_stable(file_path, check_interval=5, stable_duration=15):
        """Check if file size is stable for a period of time."""
        if not file_path.exists():
            return False
        last_size = file_path.stat().st_size
        checks = stable_duration // check_interval
        for i in range(checks):
            time.sleep(check_interval)
            if not file_path.exists():
                return False
            current_size = file_path.stat().st_size
            if current_size != last_size:
                plog(f"  Excel still changing: {last_size} -> {current_size}, waiting...")
                last_size = current_size
                return is_file_stable(file_path, check_interval, stable_duration)
        return True

    # Wait for Excel to be stable before opening
    if not is_file_stable(excel_path):
        return False, None, "Excel file is still being written"

    try:
        import openpyxl
        wb = openpyxl.load_workbook(excel_path)

        scenes_sheet = None
        for sheet_name in wb.sheetnames:
            if 'scene' in sheet_name.lower():
                scenes_sheet = wb[sheet_name]
                break

        if not scenes_sheet:
            return False, None, "No Scenes sheet in Excel"

        headers = [cell.value for cell in scenes_sheet[1]]

        id_col = start_col = None
        for i, h in enumerate(headers):
            if h is None:
                continue
            h_lower = str(h).lower().strip()
            if h_lower in ['scene_id', 'id'] and id_col is None:
                id_col = i
            if h_lower == 'srt_start':
                start_col = i
            elif 'start' in h_lower and 'time' in h_lower and start_col is None:
                start_col = i

        if id_col is None:
            return False, None, "No ID column found"

        # Load media
        media_items = []
        video_count = 0
        image_count = 0

        for row in scenes_sheet.iter_rows(min_row=2, values_only=True):
            if row[id_col] is None:
                continue

            scene_id_raw = str(row[id_col]).strip()

            try:
                scene_id_int = int(float(scene_id_raw))
                scene_id = str(scene_id_int)
            except ValueError:
                continue

            media_path = None
            is_video = False

            possible_ids = [scene_id, f"{scene_id}.0", scene_id_raw]
            possible_ids = list(dict.fromkeys(possible_ids))

            for sid in possible_ids:
                video_path = img_dir / f"{sid}.mp4"

                if video_path.exists():
                    media_path = video_path
                    is_video = True
                    video_count += 1
                    break

                # Check multiple image formats
                for img_ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    img_path = img_dir / f"{sid}{img_ext}"
                    if img_path.exists():
                        media_path = img_path
                        is_video = False
                        image_count += 1
                        break
                if media_path:
                    break

            if not media_path:
                continue

            start_time = 0.0
            if start_col is not None and row[start_col]:
                start_time = parse_timestamp(str(row[start_col]))

            media_items.append({
                'id': scene_id,
                'path': str(media_path),
                'start': start_time,
                'is_video': is_video
            })

        if not media_items:
            return False, None, "No media found in img/ folder"

        media_items.sort(key=lambda x: x['start'])
        plog(f"  Found {len(media_items)} media: {video_count} videos, {image_count} images")

        # Handle gap at start
        GAP_THRESHOLD = 0.5
        first_start = media_items[0]['start']

        if first_start > GAP_THRESHOLD:
            plog(f"  Gap at start: 0:00 -> {first_start:.1f}s, using first media as filler")
            filler_item = {
                'id': f"{media_items[0]['id']}_filler",
                'path': media_items[0]['path'],
                'start': 0.0,
                'is_video': media_items[0]['is_video'],
                'is_filler': True
            }
            media_items.insert(0, filler_item)

        # Get voice duration
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(voice_path)]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        total_duration = float(result.stdout.strip()) if result.stdout.strip() else 60.0
        plog(f"  Voice duration: {total_duration:.1f}s")

        # Calculate durations
        for i, item in enumerate(media_items):
            if i == 0:
                if len(media_items) > 1:
                    item['duration'] = media_items[1]['start']
                else:
                    item['duration'] = total_duration
            elif i < len(media_items) - 1:
                item['duration'] = media_items[i + 1]['start'] - item['start']
            else:
                item['duration'] = total_duration - item['start']

            if item['duration'] <= 0:
                item['duration'] = max(0.5, (total_duration - item['start']) / max(1, len(media_items) - i))

        # Create video
        temp_dir = tempfile.mkdtemp()
        try:
            temp_video = Path(temp_dir) / "temp_video.mp4"

            # Load settings from channel template (per-channel customization)
            channel_template = get_subtitle_template(code)
            channel = code.split("-")[0] if "-" in code else code

            # Video settings from template (with defaults)
            output_resolution = channel_template.get("output_resolution", "4k").lower()
            compose_mode = channel_template.get("compose_mode", "quality").lower()
            kb_intensity = channel_template.get("ken_burns_intensity", "subtle").lower()
            video_transition = channel_template.get("video_transition", "random").lower()
            output_fps = 30  # Fixed FPS
            transition_duration = 0.5

            # Fallback to global config if no template settings
            try:
                import yaml
                config_path = TOOL_DIR / "config" / "settings.yaml"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                    # Only use global config if not set in template
                    if "output_resolution" not in channel_template:
                        output_resolution = config.get('output_resolution', '4k').lower()
                    if "compose_mode" not in channel_template:
                        compose_mode = config.get('video_compose_mode', 'quality').lower()
                    output_fps = config.get('output_fps', 30)
                    transition_duration = config.get('transition_duration', 0.5)
            except:
                pass

            plog(f"  Channel: {channel} | Res: {output_resolution.upper()} | Mode: {compose_mode}")

            # Determine if using xfade transitions (mix/wipe need xfade filter)
            use_xfade = video_transition in ["mix", "wipe", "random"]
            FADE_DURATION = transition_duration if use_xfade else 0.4

            # Use OpenCV Ken Burns for quality/balanced modes
            use_opencv_kb = KEN_BURNS_CV2_AVAILABLE and compose_mode in ["quality", "balanced"]

            # Detect GPU for FFmpeg fallback
            use_gpu = False
            gpu_encoder = "libx264"
            try:
                gpu_check = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5, creationflags=SUBPROCESS_FLAGS)
                if "h264_nvenc" in gpu_check.stdout:
                    use_gpu = True
                    gpu_encoder = "h264_nvenc"
                    plog(f"  GPU Encoder: NVENC")
            except:
                pass

            # Initialize Ken Burns generator
            # For xfade transitions, don't apply individual clip fades (xfade handles it)
            clip_fade_duration = 0.0 if use_xfade else FADE_DURATION
            if use_opencv_kb:
                ken_burns = KenBurnsCv2(
                    output_resolution=output_resolution,
                    fps=output_fps,
                    fade_duration=clip_fade_duration,
                    intensity=kb_intensity
                )
                plog(f"  Ken Burns intensity: {kb_intensity.upper()}")
                # Determine output size
                if output_resolution == "auto":
                    # Use first media to detect
                    first_media = media_items[0]['path']
                    import cv2
                    if media_items[0]['is_video']:
                        cap = cv2.VideoCapture(str(first_media))
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                    else:
                        img = cv2.imread(str(first_media))
                        h, w = img.shape[:2]
                    output_size = ken_burns.detect_optimal_resolution(w, h)
                else:
                    output_size = QUALITY_PRESETS.get(output_resolution, QUALITY_PRESETS["1080p"])
                plog(f"  Output: {output_size[0]}x{output_size[1]} @ {output_fps}fps")
            else:
                # Fallback to FFmpeg Ken Burns
                ken_burns = KenBurnsGenerator(1920, 1080, intensity="normal")
                output_size = (1920, 1080)

            plog(f"  Compose mode: {compose_mode.upper()} ({'OpenCV' if use_opencv_kb else 'FFmpeg'})")
            plog(f"  Transition: {video_transition.upper()} ({transition_duration}s)")
            plog(f"  Creating {len(media_items)} clips with {CLIP_WORKERS} parallel workers...")
            total_clips = len(media_items)
            update_progress(step="Creating clips", percent=5, clip_total=total_clips)

            # Helper function for creating a single clip
            def create_single_clip(task_args):
                """Create a single clip - runs in parallel worker."""
                (idx, item_data, clip_path_str, kb_config) = task_args
                clip_path = Path(clip_path_str)
                media_path = Path(item_data['path'])
                target_duration = item_data['duration']
                is_video = item_data['is_video']
                success = False

                # Create worker-local Ken Burns instance for thread safety
                worker_kb = None
                if kb_config['use_opencv']:
                    try:
                        worker_kb = KenBurnsCv2(
                            output_resolution=kb_config['resolution'],
                            fps=kb_config['fps'],
                            fade_duration=kb_config['fade_duration'],
                            intensity=kb_config['intensity']
                        )
                    except:
                        pass

                if is_video:
                    # Process VIDEO clip
                    if worker_kb:
                        success = worker_kb.process_video_clip(
                            media_path, clip_path, target_duration, kb_config['output_size']
                        )

                    if not success:
                        # Fallback to FFmpeg
                        abs_path = str(media_path.resolve()).replace('\\', '/')
                        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "default=noprint_wrappers=1:nokey=1", abs_path]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
                        video_duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 8.0

                        out_w, out_h = kb_config['output_size']
                        base_vf = f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
                        if kb_config['use_xfade']:
                            vf = base_vf
                        else:
                            vf = f"{base_vf},fade=t=in:st=0:d={kb_config['fade_dur']},fade=t=out:st={max(0, target_duration - kb_config['fade_dur'])}:d={kb_config['fade_dur']}"

                        v_encoder = kb_config['gpu_encoder'] if kb_config['use_gpu'] else "libx264"
                        v_preset = ["-preset", "p4"] if kb_config['use_gpu'] else ["-preset", "medium"]

                        if video_duration > target_duration:
                            trim_start = (video_duration - target_duration) / 2
                            cmd_clip = [
                                "ffmpeg", "-y", "-ss", str(trim_start), "-i", abs_path,
                                "-t", str(target_duration), "-vf", vf,
                                "-c:v", v_encoder, *v_preset, "-pix_fmt", "yuv420p",
                                "-an", "-r", str(kb_config['fps']), str(clip_path)
                            ]
                        else:
                            cmd_clip = [
                                "ffmpeg", "-y", "-i", abs_path, "-t", str(target_duration),
                                "-vf", vf, "-c:v", v_encoder, *v_preset,
                                "-pix_fmt", "yuv420p", "-an", "-r", str(kb_config['fps']), str(clip_path)
                            ]

                        result = subprocess.run(cmd_clip, capture_output=True, text=True, timeout=300, creationflags=SUBPROCESS_FLAGS)
                        success = result.returncode == 0
                else:
                    # Process IMAGE with Ken Burns effect
                    if worker_kb:
                        success = worker_kb.create_clip_from_image(
                            media_path, clip_path, target_duration,
                            effect=None,
                            output_size=kb_config['output_size']
                        )

                    if not success:
                        # Fallback to FFmpeg
                        abs_path = str(media_path.resolve()).replace('\\', '/')
                        out_w, out_h = kb_config['output_size']
                        base_filter = f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
                        if kb_config['use_xfade']:
                            vf = base_filter
                        else:
                            vf = f"{base_filter},fade=t=in:st=0:d={kb_config['fade_dur']},fade=t=out:st={max(0, target_duration - kb_config['fade_dur'])}:d={kb_config['fade_dur']}"

                        if kb_config['use_gpu']:
                            cmd_clip = [
                                "ffmpeg", "-y", "-loop", "1", "-t", str(target_duration),
                                "-i", abs_path, "-vf", vf, "-c:v", kb_config['gpu_encoder'],
                                "-preset", "p4", "-rc", "vbr", "-cq", "22",
                                "-pix_fmt", "yuv420p", "-r", str(kb_config['fps']), str(clip_path)
                            ]
                        else:
                            cpu_preset = "ultrafast" if kb_config['compose_mode'] == "fast" else "medium"
                            cmd_clip = [
                                "ffmpeg", "-y", "-loop", "1", "-t", str(target_duration),
                                "-i", abs_path, "-vf", vf, "-c:v", "libx264",
                                "-preset", cpu_preset, "-pix_fmt", "yuv420p", "-r", str(kb_config['fps']), str(clip_path)
                            ]

                        result = subprocess.run(cmd_clip, capture_output=True, text=True, timeout=300, creationflags=SUBPROCESS_FLAGS)
                        success = result.returncode == 0

                return (idx, success, str(clip_path) if success and clip_path.exists() else None)

            # Prepare config for workers
            kb_config = {
                'use_opencv': use_opencv_kb,
                'resolution': output_resolution,
                'fps': output_fps,
                'fade_duration': clip_fade_duration,
                'intensity': kb_intensity,
                'output_size': output_size,
                'use_xfade': use_xfade,
                'fade_dur': FADE_DURATION,
                'use_gpu': use_gpu,
                'gpu_encoder': gpu_encoder,
                'compose_mode': compose_mode,
            }

            # Create task list
            clip_tasks = []
            for i, item in enumerate(media_items):
                clip_path = Path(temp_dir) / f"clip_{i:03d}.mp4"
                clip_tasks.append((i, item, str(clip_path), kb_config))

            # Process clips in parallel
            clip_results = [None] * total_clips
            completed_count = 0

            with ThreadPoolExecutor(max_workers=CLIP_WORKERS) as executor:
                futures = {executor.submit(create_single_clip, task): task[0] for task in clip_tasks}

                for future in as_completed(futures):
                    idx, success, clip_path_str = future.result()
                    if success and clip_path_str:
                        clip_results[idx] = Path(clip_path_str)

                    completed_count += 1
                    clip_percent = 5 + int(completed_count / total_clips * 65)
                    update_progress(clip_current=completed_count, percent=clip_percent)

                    if completed_count % 20 == 0:
                        plog(f"  ... {completed_count}/{total_clips} clips")

            # Collect successful clips in order
            clip_paths = [p for p in clip_results if p is not None]

            if not clip_paths:
                return False, None, "No clips created"

            plog(f"  Created {len(clip_paths)} clips...")

            # Re-encode clips for high quality xfade (avoid artifacts during crossfade)
            # Only needed when using xfade transitions
            if use_xfade and len(clip_paths) > 1:
                plog(f"  Re-encoding clips for smooth crossfade...")
                update_progress(step="Optimizing clips", percent=72)

                reencoded_paths = []
                for i, cp in enumerate(clip_paths):
                    reenc_path = Path(temp_dir) / f"hq_{i:03d}.mp4"

                    # Re-encode with high quality, consistent format
                    if use_gpu:
                        reencode_cmd = [
                            "ffmpeg", "-y", "-i", str(cp),
                            "-c:v", gpu_encoder, "-preset", "p5",
                            "-rc", "vbr", "-cq", "18", "-b:v", "20M",
                            "-pix_fmt", "yuv420p", "-an",
                            str(reenc_path)
                        ]
                    else:
                        reencode_cmd = [
                            "ffmpeg", "-y", "-i", str(cp),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "17",
                            "-profile:v", "high", "-pix_fmt", "yuv420p", "-an",
                            str(reenc_path)
                        ]

                    result = subprocess.run(reencode_cmd, capture_output=True, text=True, timeout=120, creationflags=SUBPROCESS_FLAGS)
                    if result.returncode == 0 and reenc_path.exists():
                        reencoded_paths.append(reenc_path)
                        # Delete original to save space
                        try:
                            cp.unlink()
                        except:
                            pass
                    else:
                        # Keep original if re-encode fails
                        reencoded_paths.append(cp)

                clip_paths = reencoded_paths
                plog(f"  Re-encoded {len(clip_paths)} clips")

            update_progress(step="Concatenating", percent=75)

            # Concat with appropriate transition
            if use_xfade and len(clip_paths) > 1:
                # Use xfade filter for smooth transitions
                # Build xfade filter chain
                def get_xfade_type(transition_setting):
                    """Get xfade transition type"""
                    if transition_setting == "mix":
                        return "dissolve"
                    elif transition_setting == "wipe":
                        return random.choice(["wipeleft", "wiperight", "wipeup", "wipedown"])
                    elif transition_setting == "random":
                        # 40% fade_black, 45% mix, 15% wipe
                        r = random.random()
                        if r < 0.40:
                            return "fade"  # fade through black
                        elif r < 0.85:
                            return "dissolve"  # crossfade
                        else:
                            return random.choice(["wipeleft", "wiperight"])
                    return "dissolve"

                def xfade_batch(batch_clips, batch_idx, temp_dir_path):
                    """Process a batch of clips with xfade transitions"""
                    if len(batch_clips) == 1:
                        return batch_clips[0]  # Single clip, no xfade needed

                    # Get clip durations
                    batch_durations = []
                    for cp in batch_clips:
                        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "default=noprint_wrappers=1:nokey=1", str(cp)]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
                        dur = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 5.0
                        batch_durations.append(dur)

                    # Build inputs
                    inputs = []
                    for cp in batch_clips:
                        inputs.extend(["-i", str(cp).replace('\\', '/')])

                    # Build filter_complex
                    filter_parts = []
                    current_offset = 0
                    prev_label = "[0]"

                    for i in range(1, len(batch_clips)):
                        xfade_type = get_xfade_type(video_transition)
                        current_offset += batch_durations[i-1] - transition_duration
                        out_label = f"[v{i}]" if i < len(batch_clips) - 1 else "[vout]"
                        filter_parts.append(
                            f"{prev_label}[{i}]xfade=transition={xfade_type}:duration={transition_duration}:offset={current_offset:.3f}{out_label}"
                        )
                        prev_label = out_label

                    filter_complex = ";".join(filter_parts) + ";[vout]format=yuv420p[vfinal]"

                    # Encoder settings
                    if use_gpu:
                        v_encoder = gpu_encoder
                        v_quality = ["-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "15M", "-maxrate", "20M"]
                    else:
                        v_encoder = "libx264"
                        v_quality = ["-preset", "slow", "-crf", "18", "-profile:v", "high"]

                    batch_output = Path(temp_dir_path) / f"batch_{batch_idx:03d}.mp4"
                    cmd_batch = ["ffmpeg", "-y"] + inputs + [
                        "-filter_complex", filter_complex,
                        "-map", "[vfinal]",
                        "-c:v", v_encoder, *v_quality,
                        "-pix_fmt", "yuv420p",
                        "-r", str(output_fps),
                        str(batch_output)
                    ]

                    result = subprocess.run(cmd_batch, capture_output=True, text=True, timeout=600, creationflags=SUBPROCESS_FLAGS)
                    if result.returncode == 0 and batch_output.exists():
                        return batch_output
                    return None

                # Process in batches to avoid Windows command line length limit
                BATCH_SIZE = 40  # Safe batch size for Windows
                xfade_success = False

                if len(clip_paths) <= BATCH_SIZE:
                    # Small video - process all at once
                    plog(f"  Using xfade transitions ({video_transition})...")
                    batch_output = xfade_batch(clip_paths, 0, temp_dir)
                    if batch_output:
                        shutil.move(str(batch_output), str(temp_video))
                        xfade_success = True
                else:
                    # Large video - process in batches
                    plog(f"  Processing {len(clip_paths)} clips in batches of {BATCH_SIZE}...")
                    batch_outputs = []
                    for batch_idx in range(0, len(clip_paths), BATCH_SIZE):
                        batch_clips = clip_paths[batch_idx:batch_idx + BATCH_SIZE]
                        plog(f"    Batch {batch_idx // BATCH_SIZE + 1}: clips {batch_idx + 1}-{batch_idx + len(batch_clips)}")
                        batch_output = xfade_batch(batch_clips, batch_idx // BATCH_SIZE, temp_dir)
                        if batch_output:
                            batch_outputs.append(batch_output)
                        else:
                            plog(f"    Batch {batch_idx // BATCH_SIZE + 1} failed", "WARN")
                            break

                    if len(batch_outputs) == (len(clip_paths) + BATCH_SIZE - 1) // BATCH_SIZE:
                        # All batches successful - concat them
                        plog(f"  Combining {len(batch_outputs)} batches...")
                        batch_list = Path(temp_dir) / "batches.txt"
                        with open(batch_list, 'w', encoding='utf-8') as f:
                            for bp in batch_outputs:
                                f.write(f"file '{str(bp).replace(chr(92), '/')}'\n")
                        cmd_final = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(batch_list), "-c", "copy", str(temp_video)]
                        result = subprocess.run(cmd_final, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
                        xfade_success = result.returncode == 0

                if not xfade_success:
                    plog(f"  xfade failed, falling back to simple concat", "WARN")
                    # Fallback to simple concat
                    list_file = Path(temp_dir) / "clips.txt"
                    with open(list_file, 'w', encoding='utf-8') as f:
                        for cp in clip_paths:
                            f.write(f"file '{str(cp).replace(chr(92), '/')}'\n")
                    cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(temp_video)]
                    result = subprocess.run(cmd_concat, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
                    if result.returncode != 0:
                        return False, None, f"Concat error: {result.stderr[-200:]}"
            else:
                # Simple concat (for fade_black or single clip)
                list_file = Path(temp_dir) / "clips.txt"
                with open(list_file, 'w', encoding='utf-8') as f:
                    for cp in clip_paths:
                        f.write(f"file '{str(cp).replace(chr(92), '/')}'\n")

                cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(temp_video)]
                result = subprocess.run(cmd_concat, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
                if result.returncode != 0:
                    return False, None, f"Concat error: {result.stderr[-200:]}"

            # Add audio
            temp_with_audio = Path(temp_dir) / "with_audio.mp4"
            update_progress(step="Adding audio", percent=85)
            plog("  Adding voice...")
            cmd2 = ["ffmpeg", "-y", "-i", str(temp_video), "-i", str(voice_path),
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(temp_with_audio)]
            result = subprocess.run(cmd2, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
            if result.returncode != 0:
                return False, None, f"Audio merge error: {result.stderr[-200:]}"

            # Burn subtitles
            temp_with_subs = Path(temp_dir) / "with_subs.mp4"
            if srt_path and srt_path.exists():
                update_progress(step="Burning subtitles", percent=90)
                plog("  Burning subtitles...")
                srt_escaped = str(srt_path).replace('\\', '/').replace(':', '\\:')

                # Use local fonts from fonts/ folder with channel template
                fonts_dir = str(TOOL_DIR / "fonts").replace('\\', '/').replace(':', '\\:')
                template = get_subtitle_template(code)
                subtitle_style = (
                    f"FontName={template['font']},FontSize={template['size']},"
                    f"PrimaryColour={template['color']},OutlineColour={template['outline']},"
                    f"BorderStyle=1,Outline={template['outline_size']},Shadow=1,"
                    f"MarginV={template['margin_v']},Alignment={template['alignment']}"
                )
                vf_filter = f"subtitles='{srt_escaped}':fontsdir='{fonts_dir}':force_style='{subtitle_style}'"

                cmd3 = ["ffmpeg", "-y", "-i", str(temp_with_audio), "-vf", vf_filter, "-c:a", "copy", str(temp_with_subs)]
                result = subprocess.run(cmd3, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
                if result.returncode != 0:
                    plog(f"  Subtitle burn failed: {result.stderr[-100:]}", "WARN")
                    # Fallback: copy without subtitles
                    shutil.copy(temp_with_audio, temp_with_subs)
            else:
                shutil.copy(temp_with_audio, temp_with_subs)

            # Overlay NV image (character card) if enabled in template
            nv_enabled = channel_template.get("nv_overlay_enabled", True)
            nv_position = channel_template.get("nv_overlay_position", "left")
            nv_v_position = channel_template.get("nv_overlay_v_position", "middle")
            nv_scale = channel_template.get("nv_overlay_scale", 0.50)
            nv_crop_ratio = channel_template.get("nv_crop_ratio", 0.5)

            nv_path = find_nv_image(code, project_dir) if nv_enabled else None
            if nv_path:
                update_progress(step="Adding NV overlay", percent=95)
                plog(f"  Adding NV overlay: {nv_path.name} ({nv_position}-{nv_v_position}, {int(nv_scale*100)}%, crop={nv_crop_ratio})")
                if overlay_nv_on_video(temp_with_subs, nv_path, output_path,
                                       position=nv_position, v_position=nv_v_position,
                                       scale=nv_scale, margin=20, crop_ratio=nv_crop_ratio,
                                       callback=callback):
                    plog("  NV overlay applied successfully")
                else:
                    # Fallback: use video without NV overlay
                    shutil.copy(temp_with_subs, output_path)
            else:
                # No NV image or disabled, just copy the video with subtitles
                shutil.copy(temp_with_subs, output_path)

            update_progress(step="Done", percent=100, status="completed")
            plog(f"  Video done: {output_path.name}", "OK")
            return True, output_path, None

        finally:
            gc.collect()
            time.sleep(1)
            for attempt in range(5):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=False)
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(1)
                    if attempt == 4:
                        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        plog(f"  Video compose error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False, None, str(e)


# ============================================================================
# COPY TO DONE
# ============================================================================

def find_thumbnail(code: str) -> Optional[Path]:
    if not THUMB_DIR.exists():
        return None
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        thumb = THUMB_DIR / f"{code}{ext}"
        if thumb.exists():
            return thumb
    return None


def find_nv_image(code: str, project_dir: Path) -> Optional[Path]:
    """Find NV image (character card) for overlay."""
    # Check in VISUAL project folder first
    nv_in_project = project_dir / f"{code}_nv.png"
    if nv_in_project.exists():
        return nv_in_project

    # Check in thumb/nv folder
    nv_in_thumb = TOOL_DIR / "thumb" / "nv" / f"{code}.png"
    if nv_in_thumb.exists():
        return nv_in_thumb

    return None


def overlay_nv_on_video(video_path: Path, nv_path: Path, output_path: Path,
                        position: str = "left", v_position: str = "middle",
                        scale: float = 0.50, margin: int = 20,
                        crop_ratio: float = 0.5, callback=None) -> bool:
    """
    Overlay NV image on video.

    Args:
        video_path: Input video
        nv_path: NV image (character card with name badge)
        output_path: Output video with overlay
        position: "left" or "right" (horizontal)
        v_position: "top", "middle", or "bottom" (vertical)
        scale: Scale factor for NV image (0.50 = 50% of video height)
        margin: Margin from edge in pixels
        crop_ratio: Crop right portion of NV image (0.5 = right half, 1.0 = full image)
    """
    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(msg, level)

    try:
        # Get video dimensions
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True,
                               creationflags=SUBPROCESS_FLAGS)
        if result.returncode != 0:
            return False

        dimensions = result.stdout.strip()
        if 'x' not in dimensions:
            return False

        vid_w, vid_h = map(int, dimensions.split('x'))

        # Calculate NV overlay size (scale relative to video height)
        nv_height = int(vid_h * scale)

        # Build filter: scale NV, then overlay
        nv_escaped = str(nv_path).replace('\\', '/').replace(':', '\\:')

        # Horizontal position
        if position == "left":
            x_pos = margin
        else:
            x_pos = f"W-w-{margin}"

        # Vertical position
        if v_position == "top":
            y_pos = margin
        elif v_position == "middle":
            y_pos = "(H-h)/2"
        else:  # bottom
            y_pos = f"H-h-{margin}"

        # Filter: crop right portion (if needed), scale NV to height, then overlay
        if crop_ratio < 1.0:
            # Crop to right portion: crop=width:height:x:y
            crop_width = f"iw*{crop_ratio}"
            crop_x = f"iw*{1.0 - crop_ratio}"
            crop_filter = f"crop={crop_width}:ih:{crop_x}:0,"
        else:
            crop_filter = ""

        filter_complex = (
            f"[1:v]{crop_filter}scale=-1:{nv_height}[nv];"
            f"[0:v][nv]overlay={x_pos}:{y_pos}"
        )

        # Detect GPU encoder
        use_nvenc = False
        try:
            gpu_check = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5, creationflags=SUBPROCESS_FLAGS)
            use_nvenc = "h264_nvenc" in gpu_check.stdout
        except:
            pass

        if use_nvenc:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(nv_path),
                "-filter_complex", filter_complex,
                "-c:a", "copy",
                "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20",
                str(output_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(nv_path),
                "-filter_complex", filter_complex,
                "-c:a", "copy",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                str(output_path)
            ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, creationflags=SUBPROCESS_FLAGS)

        if result.returncode == 0:
            plog(f"  NV overlay added: {position}-{v_position}")
            return True
        else:
            plog(f"  NV overlay failed: {result.stderr[-100:]}", "WARN")
            return False

    except Exception as e:
        plog(f"  NV overlay error: {e}", "WARN")
        return False


def copy_to_done(project_info: Dict, video_path: Path, callback=None) -> Tuple[bool, Optional[str]]:
    code = project_info["code"]
    project_dir = project_info["path"]

    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(f"[{code}] {msg}", level)

    done_folder = DONE_DIR / code

    if done_folder.exists():
        plog("Removing old done folder...")
        shutil.rmtree(done_folder)

    done_folder.mkdir(parents=True, exist_ok=True)
    plog(f"Created: {done_folder}")

    # 1. Copy video
    dst_video = done_folder / video_path.name
    shutil.copy2(video_path, dst_video)
    plog(f"Copied video: {dst_video.name}")

    # 2. Copy SRT
    srt_path = project_info.get("srt_path")
    if srt_path and srt_path.exists():
        dst_srt = done_folder / f"{code}.srt"
        shutil.copy2(srt_path, dst_srt)
        plog(f"Copied SRT: {dst_srt.name}")

    # 3. Copy thumbnail (check multiple locations)
    thumb_copied = False
    # Check VISUAL folder first (generated by run_thumb.py)
    thumb_in_visual = project_dir / f"{code}.jpg"
    if thumb_in_visual.exists():
        dst_thumb = done_folder / f"{code}_thumb.jpg"
        shutil.copy2(thumb_in_visual, dst_thumb)
        plog(f"Copied thumbnail: {dst_thumb.name}")
        thumb_copied = True

    # Check thumb/thumbnails folder
    if not thumb_copied:
        thumb_in_tool = TOOL_DIR / "thumb" / "thumbnails" / f"{code}.jpg"
        if thumb_in_tool.exists():
            dst_thumb = done_folder / f"{code}_thumb.jpg"
            shutil.copy2(thumb_in_tool, dst_thumb)
            plog(f"Copied thumbnail: {dst_thumb.name}")
            thumb_copied = True

    # Fallback to global THUMB_DIR
    if not thumb_copied:
        thumb_path = find_thumbnail(code)
        if thumb_path:
            dst_thumb = done_folder / f"{code}_thumb{thumb_path.suffix}"
            shutil.copy2(thumb_path, dst_thumb)
            plog(f"Copied thumbnail: {dst_thumb.name}")

    files = list(done_folder.iterdir())
    plog(f"Done folder has {len(files)} files: {', '.join(f.name for f in files)}")

    return True, None


def delete_visual_project(project_info: Dict, callback=None) -> bool:
    code = project_info["code"]
    project_dir = project_info["path"]

    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(f"[{code}] {msg}", level)

    if not project_dir.exists():
        return True

    try:
        shutil.rmtree(project_dir)
        plog(f"Deleted VISUAL folder: {project_dir.name}")
        return True
    except Exception as e:
        plog(f"Cannot delete VISUAL folder: {e}", "WARN")
        return False


def cleanup_source_data(code: str, callback=None) -> bool:
    """Clean up source data after video is complete.

    Deletes:
    - Voice files from D:/AUTO/voice/{code}.*
    - PROJECTS/{code}/ folder
    """
    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(f"[{code}] {msg}", level)

    deleted_count = 0

    # Delete from PROJECTS folder
    projects_dir = PROJECTS_DIR / code
    if projects_dir.exists():
        try:
            shutil.rmtree(projects_dir)
            plog(f"Deleted PROJECTS folder: {code}")
            deleted_count += 1
        except Exception as e:
            plog(f"Cannot delete PROJECTS folder: {e}", "WARN")

    # Delete from voice folder (files matching {code}.*)
    if VOICE_DIR.exists():
        for item in VOICE_DIR.iterdir():
            if item.name.startswith(code + ".") or item.name == code:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    plog(f"Deleted voice file: {item.name}")
                    deleted_count += 1
                except Exception as e:
                    plog(f"Cannot delete voice file {item.name}: {e}", "WARN")

        # Also check subfolders in voice (for voice/{code}/ structure)
        voice_subdir = VOICE_DIR / code
        if voice_subdir.exists() and voice_subdir.is_dir():
            try:
                shutil.rmtree(voice_subdir)
                plog(f"Deleted voice subfolder: {code}")
                deleted_count += 1
            except Exception as e:
                plog(f"Cannot delete voice subfolder: {e}", "WARN")

    if deleted_count > 0:
        plog(f"Cleanup complete: {deleted_count} items deleted")

    return deleted_count > 0


# ============================================================================
# GOOGLE SHEET UPDATE
# ============================================================================

def load_gsheet_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        log("gspread not installed", "ERROR")
        return None, None, None

    if not CONFIG_FILE.exists():
        return None, None, None

    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        sa_path = cfg.get("SERVICE_ACCOUNT_JSON") or cfg.get("service_account_json")
        if not sa_path:
            return None, None, None

        spreadsheet_name = cfg.get("SPREADSHEET_NAME")
        if not spreadsheet_name:
            return None, None, None

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ]

        sa_file = Path(sa_path)
        if not sa_file.exists():
            sa_file = TOOL_DIR / "config" / sa_path

        if not sa_file.exists():
            return None, None, None

        creds = Credentials.from_service_account_file(str(sa_file), scopes=scopes)
        gc = gspread.authorize(creds)

        return gc, spreadsheet_name, cfg
    except Exception as e:
        log(f"Error loading gsheet client: {e}", "ERROR")
        return None, None, None


def update_sheet_status(codes: List[str], callback=None) -> Tuple[int, int]:
    if not codes:
        return 0, 0

    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(msg, level)

    gc, spreadsheet_name, cfg = load_gsheet_client()
    if not gc:
        return 0, 0

    try:
        from gspread.exceptions import APIError

        def do_update():
            ws = gc.open(spreadsheet_name).worksheet(SOURCE_SHEET_NAME)
            raw_g = ws.col_values(SOURCE_COL_CODE)
            raw_m = ws.col_values(SOURCE_COL_STATUS)

            code_to_rows = {}
            for idx, val in enumerate(raw_g, start=1):
                norm = normalize_code(val)
                if norm:
                    code_to_rows.setdefault(norm, []).append(idx)

            targets = [normalize_code(c) for c in codes if c]
            targets = list(set(t for t in targets if t))

            plog(f"Updating {len(targets)} codes in sheet...")

            found, updates = 0, []
            for code in targets:
                rows = code_to_rows.get(code, [])
                if not rows:
                    continue

                found += len(rows)
                for r in rows:
                    current = raw_m[r-1] if r-1 < len(raw_m) else ""
                    if current.strip().upper() == STATUS_VALUE.upper():
                        continue

                    plog(f"  Updating {code} @ row {r}")
                    updates.append({"range": f"M{r}", "values": [[STATUS_VALUE]]})

            if not updates:
                return found, 0

            ws.batch_update(updates, value_input_option="USER_ENTERED")
            plog(f"Updated {len(updates)} rows")
            return found, len(updates)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return do_update()
            except APIError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    plog(f"API error {e.response.status_code}, retrying in {delay}s...", "WARN")
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                last_error = e
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    plog(f"Network error, retrying in {delay}s...", "WARN")
                    time.sleep(delay)
                else:
                    raise

        raise last_error

    except Exception as e:
        plog(f"Error updating sheet: {e}", "ERROR")
        return 0, 0


# ============================================================================
# PROCESS PROJECT
# ============================================================================

def generate_thumbnail_for_project(project_info: Dict, callback=None) -> bool:
    """Generate thumbnail for project if thumb folder exists."""
    code = project_info["code"]
    project_dir = project_info["path"]

    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(f"[{code}] {msg}", level)

    # Check for thumb folder in VISUAL project
    thumb_folder = project_dir / "thumb"
    if not thumb_folder.exists():
        return False

    # Find source image in thumb folder
    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    source_img = None
    for f in thumb_folder.iterdir():
        if f.is_file() and f.suffix.lower() in valid_ext:
            source_img = f
            break

    if not source_img:
        plog("  No source image in thumb folder", "WARN")
        return False

    plog(f"  Found thumb source: {source_img.name}")

    # Run thumbnail generation
    try:
        thumb_script = TOOL_DIR / "run_thumb.py"
        if not thumb_script.exists():
            plog("  run_thumb.py not found", "WARN")
            return False

        import subprocess
        result = subprocess.run(
            [sys.executable, str(thumb_script), code],
            capture_output=True, text=True, timeout=300,
            creationflags=SUBPROCESS_FLAGS if sys.platform == "win32" else 0
        )

        if result.returncode == 0:
            plog("  Thumbnail generated successfully")

            # Copy generated thumbnail to DONE folder
            thumb_output = TOOL_DIR / "thumb" / "thumbnails" / f"{code}.jpg"
            if thumb_output.exists():
                done_thumb = DONE_DIR / code / f"{code}_thumb.jpg"
                if (DONE_DIR / code).exists():
                    shutil.copy2(thumb_output, done_thumb)
                    plog(f"  Copied thumbnail to DONE: {done_thumb.name}")

            return True
        else:
            plog(f"  Thumbnail generation failed: {result.stderr[-200:] if result.stderr else 'Unknown error'}", "WARN")
            return False

    except subprocess.TimeoutExpired:
        plog("  Thumbnail generation timed out", "WARN")
        return False
    except Exception as e:
        plog(f"  Thumbnail error: {e}", "WARN")
        return False


def process_project(project_info: Dict, callback=None) -> bool:
    code = project_info["code"]

    def plog(msg, level="INFO"):
        if callback:
            callback(msg, level)
        else:
            log(f"[{code}] {msg}", level)

    plog("="*50)
    plog(f"Processing: {code}")
    plog("="*50)
    plog(f"Media: {project_info['video_count']} videos + {project_info['image_count']} images")
    plog(f"Scenes: {project_info['total_scenes']}")

    success, video_path, error = compose_video(project_info, callback)
    if not success:
        plog(f"Compose failed: {error}", "ERROR")
        return False

    success, error = copy_to_done(project_info, video_path, callback)
    if not success:
        plog(f"Copy failed: {error}", "ERROR")
        return False

    # Generate thumbnail if thumb folder exists
    generate_thumbnail_for_project(project_info, callback)

    delete_visual_project(project_info, callback)

    # Update sheet status with aggressive retry (MUST succeed before cleanup)
    sheet_updated = False
    max_sheet_retries = 10
    for sheet_attempt in range(max_sheet_retries):
        found, updated = update_sheet_status([code], callback)
        if updated > 0:
            plog(f"Sheet updated: {STATUS_VALUE}")
            sheet_updated = True
            break
        elif found > 0:
            # Found but already had correct status
            plog(f"Sheet already has correct status")
            sheet_updated = True
            break
        else:
            # Not found or error - retry
            if sheet_attempt < max_sheet_retries - 1:
                delay = min(30, 5 * (sheet_attempt + 1))  # 5s, 10s, 15s... max 30s
                plog(f"Sheet update failed, retrying in {delay}s... (attempt {sheet_attempt + 1}/{max_sheet_retries})", "WARN")
                time.sleep(delay)

    if not sheet_updated:
        plog(f"CRITICAL: Sheet update failed after {max_sheet_retries} attempts! NOT cleaning up source data.", "ERROR")
        plog(f"Please manually update sheet for code: {code}", "ERROR")
        # Still return True because video is done, but don't cleanup
        return True

    # Clean up source data (voice folder + PROJECTS folder) ONLY after sheet is updated
    cleanup_source_data(code, callback)

    plog(f"DONE: {code}")
    return True


# ============================================================================
# MAIN
# ============================================================================

def run_scan_loop(parallel: int = DEFAULT_PARALLEL):
    log("="*60)
    log("  VE3 TOOL - EDIT MODE (Compose MP4)")
    log("="*60)
    log(f"  VISUAL folder: {VISUAL_DIR}")
    log(f"  DONE folder:   {DONE_DIR}")
    log(f"  Parallel:      {parallel}")
    log(f"  Scan interval: {SCAN_INTERVAL}s")
    log("="*60)

    # Reset progress on start
    update_progress(code="", step="Waiting", percent=0, clip_current=0, clip_total=0, status="idle")

    DONE_DIR.mkdir(parents=True, exist_ok=True)

    cycle = 0

    while True:
        cycle += 1
        log(f"\n[CYCLE {cycle}] Scanning VISUAL folder...")

        pending = scan_visual_projects()

        if not pending:
            log("  No pending projects")
            update_progress(code="", step="Waiting", percent=0, clip_current=0, clip_total=0, status="idle")
        else:
            log(f"  Found {len(pending)} projects ready to edit:")
            for p in pending[:5]:
                log(f"    - {p['code']} ({p['video_count']}v + {p['image_count']}i / {p['total_scenes']} scenes)")
            if len(pending) > 5:
                log(f"    ... and {len(pending) - 5} more")

            batch = pending[:parallel]
            log(f"\n  Processing {len(batch)} projects...")

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(process_project, p): p for p in batch}

                for future in as_completed(futures):
                    project = futures[future]
                    try:
                        success = future.result()
                        if success:
                            log(f"  {project['code']}: SUCCESS", "OK")
                        else:
                            log(f"  {project['code']}: FAILED", "ERROR")
                    except Exception as e:
                        log(f"  {project['code']}: EXCEPTION - {e}", "ERROR")

            # Reset progress after batch
            update_progress(code="", step="Waiting", percent=0, clip_current=0, clip_total=0, status="idle")

            # Immediately scan again if there were projects (continuous processing)
            log("  Batch done, scanning for more...")
            continue

        log(f"\n  Waiting {SCAN_INTERVAL}s... (Ctrl+C to stop)")
        try:
            time.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            log("\n\nStopped by user.")
            break


def run_single_project(code: str):
    project_dir = VISUAL_DIR / code

    if not project_dir.exists():
        log(f"Project not found: {project_dir}", "ERROR")
        return

    info = get_project_info(project_dir)

    if info["already_done"]:
        log(f"Project already done: {code}", "WARN")
        return

    if not info["ready_for_edit"]:
        log(f"Project not ready: {code}", "WARN")
        log(f"  Media: {info['video_count']}v + {info['image_count']}i / {info['total_scenes']} scenes")
        log(f"  Audio: {info['has_audio']}")
        log(f"  Excel: {info['has_excel']}")
        return

    process_project(info)


def run_scan_only():
    log("="*60)
    log("  VE3 TOOL - EDIT MODE (Scan Only)")
    log("="*60)

    pending = scan_visual_projects()

    if not pending:
        log("No pending projects found")
        return

    log(f"\nFound {len(pending)} projects ready to edit:\n")

    for p in pending:
        log(f"  {p['code']}:")
        log(f"    Media:  {p['video_count']} videos + {p['image_count']} images / {p['total_scenes']} scenes")
        log(f"    Audio:  {'YES' if p['has_audio'] else 'NO'}")
        log(f"    Excel:  {'YES' if p['has_excel'] else 'NO'}")
        log(f"    SRT:    {'YES' if p['has_srt'] else 'NO'}")


def main():
    parser = argparse.ArgumentParser(description="VE3 Tool - Edit Mode (Compose MP4)")
    parser.add_argument("code", nargs="?", help="Process single project by code")
    parser.add_argument("--parallel", "-p", type=int, default=DEFAULT_PARALLEL,
                        help=f"Number of parallel workers (default: {DEFAULT_PARALLEL})")
    parser.add_argument("--scan-only", action="store_true", help="Only scan and show status")
    args = parser.parse_args()

    if args.scan_only:
        run_scan_only()
    elif args.code:
        run_single_project(args.code)
    else:
        run_scan_loop(parallel=args.parallel)


if __name__ == "__main__":
    main()
