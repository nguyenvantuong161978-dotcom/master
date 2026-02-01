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

# File stability check settings
FILE_STABLE_CHECK_INTERVAL = 5  # Check every 5 seconds
FILE_STABLE_DURATION = 30  # File must be stable for 30 seconds


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def is_file_stable(file_path: Path, check_interval: int = FILE_STABLE_CHECK_INTERVAL,
                   stable_duration: int = FILE_STABLE_DURATION) -> bool:
    """Check if file size is stable (not changing) for a period of time.

    Returns True if file hasn't changed size for stable_duration seconds.
    Returns False if file is still being written/downloaded.
    """
    if not file_path.exists():
        return False

    checks_needed = stable_duration // check_interval
    last_size = file_path.stat().st_size

    for i in range(checks_needed):
        time.sleep(check_interval)

        if not file_path.exists():
            return False

        current_size = file_path.stat().st_size

        if current_size != last_size:
            # File is still changing, reset and continue monitoring
            safe_print(f"  [WAIT] {file_path.name}: size changed {last_size} -> {current_size}, waiting...")
            last_size = current_size
            # Reset counter - start fresh monitoring
            return is_file_stable(file_path, check_interval, stable_duration)

        # Show progress
        elapsed = (i + 1) * check_interval
        safe_print(f"  [WAIT] {file_path.name}: stable for {elapsed}/{stable_duration}s...")

    return True


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
        srt_path = Path(srt_path)

        # Handle .srt.tmp -> .txt.tmp
        if srt_path.name.endswith('.srt.tmp'):
            txt_path = srt_path.parent / (srt_path.stem.replace('.srt', '') + '.txt.tmp')
        else:
            txt_path = srt_path.with_suffix(".txt")

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
                # Retry up to 3 times with delay for locked files
                for attempt in range(3):
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        print(f"  Deleted: {item.name}")
                        deleted_count += 1
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(1)  # Wait 1s before retry
                        else:
                            print(f"  Cannot delete {item.name}: file in use (will retry later)")
                    except Exception as e:
                        print(f"  Cannot delete {item.name}: {e}")
                        break

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


def copy_to_projects_atomic(name: str, voice_path: Path, local_srt: Path, local_txt: Path = None):
    """Copy files to PROJECTS atomically using temp folder.

    This prevents VM from seeing incomplete project folder:
    1. Create PROJECTS/name.tmp/ folder
    2. Copy all files there
    3. Rename to PROJECTS/name/ when complete
    """
    output_dir = PROJECTS_DIR / name
    temp_dir = PROJECTS_DIR / f"{name}.tmp"

    # Already done?
    if output_dir.exists() and (output_dir / f"{name}.srt").exists():
        return True

    try:
        # Clean up any existing temp folder
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        # Create temp folder
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Copy SRT first (most important)
        temp_srt = temp_dir / f"{name}.srt"
        shutil.copy2(local_srt, temp_srt)

        # Copy voice
        temp_voice = temp_dir / voice_path.name
        shutil.copy2(voice_path, temp_voice)

        # Copy txt if exists
        if local_txt and local_txt.exists():
            temp_txt = temp_dir / f"{name}.txt"
            shutil.copy2(local_txt, temp_txt)

        # Remove existing output folder if any
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # Atomic rename: temp -> final (VM will only see complete folder)
        temp_dir.rename(output_dir)

        safe_print(f"[SRT] {name}: Copied to PROJECTS (atomic)")
        return True

    except Exception as e:
        safe_print(f"[SRT] {name}: Copy to PROJECTS failed - {e}")
        # Cleanup temp folder
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        return False


def process_voice_to_srt(voice_path: Path) -> bool:
    """Process voice file to SRT.

    Flow:
    1. Create SRT in voice folder (as .srt.tmp while processing)
    2. When Whisper completes, rename .tmp to .srt
    3. Copy voice + SRT to PROJECTS atomically (using temp folder)
    4. Delete source voice files

    This ensures:
    - SRT only appears when fully complete
    - VM only sees PROJECTS folder when all files are ready
    """
    name = voice_path.stem
    voice_dir = voice_path.parent
    is_from_source = PROJECTS_DIR not in voice_path.parents

    # Paths in voice folder (where we create SRT first)
    local_srt_tmp = voice_dir / f"{name}.srt.tmp"
    local_srt = voice_dir / f"{name}.srt"
    local_txt = voice_dir / f"{name}.txt"

    # Paths in PROJECTS folder (final destination)
    output_dir = PROJECTS_DIR / name
    project_srt = output_dir / f"{name}.srt"

    # Already done in PROJECTS?
    if project_srt.exists():
        return True

    # Already have local SRT? Just copy to PROJECTS atomically
    if local_srt.exists():
        safe_print(f"[SRT] {name}: Local SRT found, copying to PROJECTS...")

        # Check if .txt.tmp exists (leftover from failed previous run)
        local_txt_tmp = voice_dir / f"{name}.txt.tmp"
        if local_txt_tmp.exists():
            safe_print(f"[SRT] {name}: Found leftover .txt.tmp, renaming...")
            try:
                if local_txt.exists():
                    local_txt.unlink()
                local_txt_tmp.rename(local_txt)
            except Exception as e:
                safe_print(f"[SRT] {name}: Cannot rename txt.tmp: {e}")

        if copy_to_projects_atomic(name, voice_path, local_srt, local_txt):
            return True
        return False

    # Check if voice file is stable (not being written/downloaded)
    safe_print(f"[SRT] {name}: Checking if file is stable...")
    if not is_file_stable(voice_path):
        safe_print(f"[SRT] {name}: File is still changing, skipping for now...")
        return False

    # Create SRT in voice folder
    safe_print(f"[SRT] {name}: File stable, creating SRT (Whisper)...")
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

        # Create SRT as .tmp first (in voice folder)
        conv.transcribe(str(voice_path), str(local_srt_tmp))

        # Rename .tmp to .srt (delete existing first if any)
        if local_srt.exists():
            local_srt.unlink()
        local_srt_tmp.rename(local_srt)

        # Also rename the .txt.tmp if it was created
        local_txt_tmp = voice_dir / f"{name}.txt.tmp"
        if local_txt_tmp.exists():
            if local_txt.exists():
                local_txt.unlink()
            local_txt_tmp.rename(local_txt)

        safe_print(f"[SRT] {name}: SRT created in voice folder")

        # Now copy to PROJECTS atomically (VM only sees when complete)
        if copy_to_projects_atomic(name, voice_path, local_srt, local_txt):
            safe_print(f"[SRT] {name}: Done!")
            return True
        return False

    except Exception as e:
        safe_print(f"[SRT] {name}: Error - {e}")
        # Cleanup tmp file if failed
        if local_srt_tmp.exists():
            try:
                local_srt_tmp.unlink()
            except:
                pass
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
    """Get voice files that need SRT or need copying to PROJECTS."""
    pending = []

    voice_files = scan_voice_folder(voice_dir)
    for voice_path in voice_files:
        name = voice_path.stem
        voice_folder = voice_path.parent

        # Check if SRT exists in PROJECTS
        project_dir = PROJECTS_DIR / name
        project_srt = project_dir / f"{name}.srt"

        # Check if local SRT exists (in voice folder)
        local_srt = voice_folder / f"{name}.srt"

        # Need processing if: no project SRT (either need Whisper or need copy)
        if not project_srt.exists():
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
