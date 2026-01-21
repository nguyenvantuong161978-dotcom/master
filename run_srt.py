#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Tool - MASTER: Voice to SRT
Tự động quét thư mục voice và tạo SRT bằng Whisper.

Usage:
    python run_srt.py                     (quét D:\AUTO\voice mặc định)
    python run_srt.py D:\path\to\voice    (quét thư mục khác)
"""

import sys
import os
import shutil
import time
import threading
import re
import logging
from pathlib import Path
from datetime import timedelta

# ============================================================================
# CONFIG
# ============================================================================

TOOL_DIR = Path(__file__).parent
DEFAULT_VOICE_DIR = Path("D:/AUTO/voice")
PROJECTS_DIR = TOOL_DIR / "PROJECTS"
SCAN_INTERVAL = 30

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def get_logger(name: str = "ve3_tool") -> logging.Logger:
    return logging.getLogger(name)


def format_srt_time(td: timedelta) -> str:
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


# ============================================================================
# VOICE TO SRT (inline from modules/voice_to_srt.py)
# ============================================================================

WHISPER_AVAILABLE = False
WHISPER_TIMESTAMPED_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    import whisper_timestamped
    WHISPER_TIMESTAMPED_AVAILABLE = True
except ImportError:
    pass


class VoiceToSrt:
    """Chuyển đổi file audio thành file SRT bằng Whisper."""

    def __init__(self, model_name: str = "medium", language: str = None, device: str = None):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.logger = get_logger("voice_to_srt")

        if not WHISPER_AVAILABLE and not WHISPER_TIMESTAMPED_AVAILABLE:
            raise RuntimeError("Whisper chưa cài! pip install openai-whisper hoặc whisper-timestamped")

        self.use_timestamped = WHISPER_TIMESTAMPED_AVAILABLE
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        print(f"  Loading Whisper model '{self.model_name}'...")

        if self.use_timestamped:
            import whisper_timestamped
            self._model = whisper_timestamped.load_model(self.model_name, device=self.device)
        else:
            import whisper
            self._model = whisper.load_model(self.model_name, device=self.device)

        print(f"  Whisper model loaded!")

    def transcribe(self, input_audio_path, output_srt_path, **kwargs):
        input_audio_path = Path(input_audio_path)
        output_srt_path = Path(output_srt_path)

        if not input_audio_path.exists():
            raise FileNotFoundError(f"File audio không tồn tại: {input_audio_path}")

        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_model()

        print(f"  Transcribing audio...")

        if self.use_timestamped:
            result = self._transcribe_timestamped(input_audio_path, **kwargs)
        else:
            result = self._transcribe_standard(input_audio_path, **kwargs)

        self._write_srt(result, output_srt_path)
        return result

    def _transcribe_timestamped(self, audio_path, **kwargs):
        import whisper_timestamped

        options = {
            "language": self.language,
            "beam_size": 5,
            "best_of": 5,
            "vad": True,
            "detect_disfluencies": False,
        }
        options.update(kwargs)

        try:
            result = whisper_timestamped.transcribe(self._model, str(audio_path), **options)
        except Exception as e:
            if "silero" in str(e).lower() or "vad" in str(e).lower() or "select()" in str(e):
                options["vad"] = False
                result = whisper_timestamped.transcribe(self._model, str(audio_path), **options)
            else:
                raise
        return result

    def _transcribe_standard(self, audio_path, **kwargs):
        import whisper

        options = {
            "language": self.language,
            "task": "transcribe",
            "verbose": False,
        }
        options.update(kwargs)
        return self._model.transcribe(str(audio_path), **options)

    def _write_srt(self, result, output_path):
        segments = result.get("segments", [])

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, segment in enumerate(segments, start=1):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()

                start_str = self._seconds_to_srt_time(start_time)
                end_str = self._seconds_to_srt_time(end_time)

                f.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")

        # Export TXT
        self._write_txt(result, output_path)

    def _write_txt(self, result, srt_path):
        segments = result.get("segments", [])
        txt_path = Path(srt_path).with_suffix(".txt")

        full_text = " ".join([s.get("text", "").strip() for s in segments])
        full_text = re.sub(r'([.!?])([A-ZÀ-Ỹ])', r'\1 \2', full_text)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


# ============================================================================
# SRT PROCESSING
# ============================================================================

def delete_voice_source(voice_path: Path):
    """Delete source voice files after SRT created."""
    try:
        name = voice_path.stem
        voice_dir = voice_path.parent
        parent_dir = voice_dir.parent
        deleted_count = 0

        for item in voice_dir.iterdir():
            if item.name.startswith(name):
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print(f"  Deleted: {item.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Cannot delete {item.name}: {e}")

        parent_txt = parent_dir / f"{name}.txt"
        if parent_txt.exists():
            try:
                parent_txt.unlink()
                deleted_count += 1
            except:
                pass

        if voice_dir.exists() and not list(voice_dir.iterdir()):
            voice_dir.rmdir()

    except Exception as e:
        print(f"  Cleanup warning: {e}")


def process_voice_to_srt(voice_path: Path) -> bool:
    """Process voice file to SRT."""
    name = voice_path.stem
    is_from_source = PROJECTS_DIR not in voice_path.parents

    output_dir = PROJECTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    voice_copy = output_dir / voice_path.name
    srt_path = output_dir / f"{name}.srt"

    # Copy voice
    if not voice_copy.exists():
        safe_print(f"[SRT] {name}: Copying voice...")
        shutil.copy2(voice_path, voice_copy)

    # Copy txt if exists
    txt_src = voice_path.parent / f"{name}.txt"
    txt_dst = output_dir / f"{name}.txt"
    if txt_src.exists() and not txt_dst.exists():
        shutil.copy2(txt_src, txt_dst)

    if srt_path.exists():
        if is_from_source:
            delete_voice_source(voice_path)
        return True

    safe_print(f"[SRT] {name}: Creating SRT (Whisper)...")
    try:
        # Load settings
        whisper_model = 'medium'
        whisper_lang = 'en'
        try:
            import yaml
            cfg_file = TOOL_DIR / "config" / "settings.yaml"
            if cfg_file.exists():
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                whisper_model = cfg.get('whisper_model', 'medium')
                whisper_lang = cfg.get('whisper_language', 'en')
        except:
            pass

        conv = VoiceToSrt(model_name=whisper_model, language=whisper_lang)
        conv.transcribe(str(voice_copy), str(srt_path))
        safe_print(f"[SRT] {name}: Done")

        if is_from_source:
            delete_voice_source(voice_path)

        return True
    except Exception as e:
        safe_print(f"[SRT] {name}: Error - {e}")
        return False


def scan_voice_folder(voice_dir: Path) -> list:
    """Scan voice folder for audio files."""
    voice_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
    voice_files = []

    for f in voice_dir.iterdir():
        if f.is_file() and f.suffix.lower() in voice_extensions:
            voice_files.append(f)

    for subdir in voice_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.is_file() and f.suffix.lower() in voice_extensions:
                    voice_files.append(f)

    return sorted(voice_files)


def get_pending_srt(voice_dir: Path) -> list:
    """Get voice files that need SRT."""
    pending = []

    voice_files = scan_voice_folder(voice_dir)
    for voice_path in voice_files:
        name = voice_path.stem
        project_dir = PROJECTS_DIR / name
        srt_path = project_dir / f"{name}.srt"
        if not srt_path.exists():
            pending.append(voice_path)

    if PROJECTS_DIR.exists():
        for project_dir in PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            name = project_dir.name
            srt_path = project_dir / f"{name}.srt"
            if srt_path.exists():
                continue
            voice_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
            for f in project_dir.iterdir():
                if f.is_file() and f.suffix.lower() in voice_extensions:
                    pending.append(f)
                    break

    seen = set()
    result = []
    for v in pending:
        if v.stem not in seen:
            result.append(v)
            seen.add(v.stem)
    return result


def srt_worker(voice_dir: Path, stop_event: threading.Event):
    """Worker thread for SRT generation."""
    safe_print("[SRT Worker] Started")

    while not stop_event.is_set():
        try:
            pending = get_pending_srt(voice_dir)

            if pending:
                safe_print(f"[SRT Worker] Found {len(pending)} voices needing SRT")
                for voice_path in pending:
                    if stop_event.is_set():
                        break
                    process_voice_to_srt(voice_path)

            stop_event.wait(5)
        except Exception as e:
            safe_print(f"[SRT Worker] Error: {e}")
            stop_event.wait(10)

    safe_print("[SRT Worker] Stopped")


def run_scan_loop(voice_dir: Path):
    """Run SRT worker."""
    print(f"\n{'='*60}")
    print(f"  VE3 TOOL - VOICE TO SRT")
    print(f"{'='*60}")
    print(f"  Input:  {voice_dir}")
    print(f"  Output: {PROJECTS_DIR}")
    print(f"{'='*60}")

    stop_event = threading.Event()
    srt_thread = threading.Thread(target=srt_worker, args=(voice_dir, stop_event), daemon=True)
    srt_thread.start()

    cycle = 0
    try:
        while True:
            cycle += 1
            time.sleep(SCAN_INTERVAL)

            pending_srt = get_pending_srt(voice_dir)
            safe_print(f"\n[STATUS {cycle}] SRT pending: {len(pending_srt)}")

            if not pending_srt:
                safe_print("  All voices have SRT!")

    except KeyboardInterrupt:
        safe_print("\n\nStopping...")
        stop_event.set()
        srt_thread.join(timeout=5)
        safe_print("Stopped.")


def main():
    if len(sys.argv) >= 2:
        voice_dir = Path(sys.argv[1])
    else:
        voice_dir = DEFAULT_VOICE_DIR

    if not voice_dir.exists():
        print(f"[ERROR] Voice directory not found: {voice_dir}")
        return

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    run_scan_loop(voice_dir)


if __name__ == "__main__":
    main()
