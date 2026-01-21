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

# ============================================================================
# CONFIG
# ============================================================================

TOOL_DIR = Path(__file__).parent
VISUAL_DIR = Path(r"D:\AUTO\VISUAL")
DONE_DIR = Path(r"D:\AUTO\done")
THUMB_DIR = Path(r"D:\AUTO\thumbnails")
CONFIG_FILE = TOOL_DIR / "config" / "config.json"

SCAN_INTERVAL = 60
DEFAULT_PARALLEL = 2

# Google Sheet config
SOURCE_SHEET_NAME = "NGUON"
SOURCE_COL_CODE = 7
SOURCE_COL_STATUS = 13
STATUS_VALUE = "EDIT XONG"

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2


def log(msg: str, level: str = "INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


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
    img_dir = project_dir / "img"

    if not img_dir.exists():
        log(f"  [FILL] No img folder found", "WARN")
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

    img_dir = project_dir / "img"
    if img_dir.exists():
        videos = [f for f in img_dir.glob("*.mp4")
                  if not f.stem.startswith('nv') and not f.stem.startswith('loc')]
        images = [f for f in img_dir.glob("*.png")
                  if not f.stem.startswith('nv') and not f.stem.startswith('loc')]
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

            if 0.5 <= coverage < 1.0:
                log(f"    - {code}: Coverage {coverage:.0%} < 100%, filling missing...")
                filled, still_missing = fill_missing_media(project_dir, excel_path)

                if filled > 0:
                    videos = [f for f in img_dir.glob("*.mp4")
                              if not f.stem.startswith('nv') and not f.stem.startswith('loc')]
                    images = [f for f in img_dir.glob("*.png")
                              if not f.stem.startswith('nv') and not f.stem.startswith('loc')]
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

    plog("Starting video composition...")

    if not excel_path or not excel_path.exists():
        return False, None, "Excel file not found"

    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
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
    img_dir = project_dir / "img"

    plog(f"  Voice: {voice_path.name}")
    plog(f"  SRT: {srt_path.name if srt_path else 'None'}")
    plog(f"  Excel: {excel_path.name}")

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
                img_path = img_dir / f"{sid}.png"

                if video_path.exists():
                    media_path = video_path
                    is_video = True
                    video_count += 1
                    break
                elif img_path.exists():
                    media_path = img_path
                    is_video = False
                    image_count += 1
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
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
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
            FADE_DURATION = 0.4

            # Load settings
            compose_mode = "fast"
            kb_intensity = "normal"
            try:
                import yaml
                config_path = TOOL_DIR / "config" / "settings.yaml"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                    compose_mode = config.get('video_compose_mode', 'fast').lower()
                    kb_intensity = config.get('ken_burns_intensity', 'normal')
            except:
                pass

            kb_enabled = compose_mode in ["quality", "balanced"]
            use_simple_kb = compose_mode == "balanced"

            # Detect GPU
            use_gpu = False
            gpu_encoder = "libx264"
            try:
                gpu_check = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
                if "h264_nvenc" in gpu_check.stdout:
                    use_gpu = True
                    gpu_encoder = "h264_nvenc"
                    plog(f"  GPU Encoder: NVENC")
            except:
                pass

            ken_burns = KenBurnsGenerator(1920, 1080, intensity=kb_intensity)
            last_kb_effect = None

            plog(f"  Compose mode: {compose_mode.upper()}")
            plog(f"  Creating {len(media_items)} clips...")

            clip_paths = []
            for i, item in enumerate(media_items):
                clip_path = Path(temp_dir) / f"clip_{i:03d}.mp4"
                abs_path = str(Path(item['path']).resolve()).replace('\\', '/')
                target_duration = item['duration']

                rand_val = random.random()
                if rand_val < 0.2:
                    transition_type = 'none'
                elif rand_val < 0.6:
                    transition_type = 'fade_black'
                else:
                    transition_type = 'mix'

                fade_out_start = max(0, target_duration - FADE_DURATION)

                if transition_type == 'none':
                    fade_filter = ""
                elif transition_type == 'fade_black':
                    fade_filter = f"fade=t=in:st=0:d={FADE_DURATION},fade=t=out:st={fade_out_start}:d={FADE_DURATION}"
                else:
                    fade_filter = f"fade=t=in:st=0:d={FADE_DURATION}:alpha=1,fade=t=out:st={fade_out_start}:d={FADE_DURATION}:alpha=1"

                if item['is_video']:
                    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", abs_path]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    video_duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 8.0

                    if fade_filter:
                        base_vf = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,{fade_filter}"
                    else:
                        base_vf = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"

                    v_encoder = gpu_encoder if use_gpu else "libx264"
                    v_preset = ["-preset", "p4"] if use_gpu else ["-preset", "fast"]

                    if video_duration > target_duration:
                        trim_total = video_duration - target_duration
                        trim_start = trim_total / 2
                        cmd_clip = [
                            "ffmpeg", "-y", "-ss", str(trim_start), "-i", abs_path,
                            "-t", str(target_duration), "-vf", base_vf,
                            "-c:v", v_encoder, *v_preset, "-pix_fmt", "yuv420p",
                            "-an", "-r", "25", str(clip_path)
                        ]
                    else:
                        cmd_clip = [
                            "ffmpeg", "-y", "-i", abs_path, "-t", str(target_duration),
                            "-vf", base_vf, "-c:v", v_encoder, *v_preset,
                            "-pix_fmt", "yuv420p", "-an", "-r", "25", str(clip_path)
                        ]
                else:
                    MAX_KB_DURATION = 20
                    use_kb_for_this_clip = kb_enabled and target_duration <= MAX_KB_DURATION

                    if use_kb_for_this_clip:
                        kb_effect = ken_burns.get_random_effect(exclude_last=last_kb_effect)
                        last_kb_effect = kb_effect
                        vf = ken_burns.generate_filter(kb_effect, target_duration, FADE_DURATION, simple_mode=use_simple_kb)
                    else:
                        base_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
                        vf = f"{base_filter},{fade_filter}" if fade_filter else base_filter

                    if use_gpu:
                        nvenc_preset = "p1" if compose_mode == "fast" else "p4"
                        cmd_clip = [
                            "ffmpeg", "-y", "-loop", "1", "-t", str(target_duration),
                            "-i", abs_path, "-vf", vf, "-c:v", gpu_encoder,
                            "-preset", nvenc_preset, "-pix_fmt", "yuv420p", "-r", "25", str(clip_path)
                        ]
                    else:
                        cpu_preset = "ultrafast" if compose_mode == "fast" else "fast"
                        cmd_clip = [
                            "ffmpeg", "-y", "-loop", "1", "-t", str(target_duration),
                            "-i", abs_path, "-vf", vf, "-c:v", "libx264",
                            "-preset", cpu_preset, "-pix_fmt", "yuv420p", "-r", "25", str(clip_path)
                        ]

                result = subprocess.run(cmd_clip, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    plog(f"  Clip {i} failed: {result.stderr[-200:]}", "ERROR")
                    continue

                clip_paths.append(clip_path)

                if (i + 1) % 10 == 0:
                    plog(f"  ... {i + 1}/{len(media_items)} clips")

            if not clip_paths:
                return False, None, "No clips created"

            plog(f"  Created {len(clip_paths)} clips, concatenating...")

            # Concat
            list_file = Path(temp_dir) / "clips.txt"
            with open(list_file, 'w', encoding='utf-8') as f:
                for cp in clip_paths:
                    f.write(f"file '{str(cp).replace(chr(92), '/')}'\n")

            cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(temp_video)]
            result = subprocess.run(cmd_concat, capture_output=True, text=True)
            if result.returncode != 0:
                return False, None, f"Concat error: {result.stderr[-200:]}"

            # Add audio
            temp_with_audio = Path(temp_dir) / "with_audio.mp4"
            plog("  Adding voice...")
            cmd2 = ["ffmpeg", "-y", "-i", str(temp_video), "-i", str(voice_path),
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(temp_with_audio)]
            result = subprocess.run(cmd2, capture_output=True, text=True)
            if result.returncode != 0:
                return False, None, f"Audio merge error: {result.stderr[-200:]}"

            # Burn subtitles
            if srt_path and srt_path.exists():
                plog("  Burning subtitles...")
                srt_escaped = str(srt_path).replace('\\', '/').replace(':', '\\:')
                font_dir = "C\\:/Users/admin/AppData/Local/Microsoft/Windows/Fonts"

                subtitle_style = (
                    "FontName=Anton,FontSize=32,PrimaryColour=&H00FFFFFF,"
                    "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=30,Alignment=2"
                )
                vf_filter = f"subtitles='{srt_escaped}':fontsdir='{font_dir}':force_style='{subtitle_style}'"

                cmd3 = ["ffmpeg", "-y", "-i", str(temp_with_audio), "-vf", vf_filter, "-c:a", "copy", str(output_path)]
                result = subprocess.run(cmd3, capture_output=True, text=True)
                if result.returncode != 0:
                    plog("  Subtitle burn failed, trying default font...", "WARN")
                    vf_simple = f"subtitles='{srt_escaped}':force_style='FontSize=32,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2'"
                    cmd3_simple = ["ffmpeg", "-y", "-i", str(temp_with_audio), "-vf", vf_simple, "-c:a", "copy", str(output_path)]
                    result = subprocess.run(cmd3_simple, capture_output=True, text=True)
                    if result.returncode != 0:
                        shutil.copy(temp_with_audio, output_path)
            else:
                shutil.copy(temp_with_audio, output_path)

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

    dst_video = done_folder / video_path.name
    shutil.copy2(video_path, dst_video)
    plog(f"Copied video: {dst_video.name}")

    srt_path = project_info.get("srt_path")
    if srt_path and srt_path.exists():
        dst_srt = done_folder / f"{code}.srt"
        shutil.copy2(srt_path, dst_srt)
        plog(f"Copied SRT: {dst_srt.name}")

    thumb_path = find_thumbnail(code)
    if thumb_path:
        dst_thumb = done_folder / thumb_path.name
        shutil.copy2(thumb_path, dst_thumb)
        plog(f"Copied thumbnail: {dst_thumb.name}")

    files = list(done_folder.iterdir())
    plog(f"Done folder has {len(files)} files")

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

    delete_visual_project(project_info, callback)

    found, updated = update_sheet_status([code], callback)
    if updated > 0:
        plog(f"Sheet updated: {STATUS_VALUE}")

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

    DONE_DIR.mkdir(parents=True, exist_ok=True)

    cycle = 0

    while True:
        cycle += 1
        log(f"\n[CYCLE {cycle}] Scanning VISUAL folder...")

        pending = scan_visual_projects()

        if not pending:
            log("  No pending projects")
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
