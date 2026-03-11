#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Tool - Modern GUI Manager
Chạy file này để mở GUI không hiện CMD.
"""

import sys
import os
import threading
import subprocess
import time
import signal
import json
from pathlib import Path
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("Tkinter not found!")
    sys.exit(1)

# Try to import PIL for image preview
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ============================================================================
# CONFIG
# ============================================================================

TOOL_DIR = Path(__file__).parent
VISUAL_DIR = Path(r"D:\AUTO\VISUAL")
DONE_DIR = Path(r"D:\AUTO\done")
VOICE_DIR = Path(r"D:\AUTO\voice")
PROJECTS_DIR = TOOL_DIR / "PROJECTS"
PROGRESS_FILE = TOOL_DIR / "progress.json"
QUEUE_TRACKER_FILE = TOOL_DIR / "queue_tracker.json"

# Colors - GitHub Dark Theme (must be before QueueTracker)
COLORS = {
    "bg_dark": "#0d1117",
    "bg_card": "#161b22",
    "bg_card_hover": "#1f2937",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_dim": "#7d8590",
    "accent_green": "#238636",
    "accent_green_hover": "#2ea043",
    "accent_blue": "#1f6feb",
    "accent_blue_hover": "#388bfd",
    "accent_orange": "#d29922",
    "accent_red": "#da3633",
    "accent_purple": "#8957e5",
    "success": "#3fb950",
    "warning": "#d29922",
    "error": "#f85149",
}

# ============================================================================
# SMART STATE TRACKING
# ============================================================================
# States của một mã:
# 1. voice       - Có voice file, chờ SRT
# 2. srt_done    - SRT xong, voice còn, chờ VM copy
# 3. waiting_vm  - Voice bị xóa (VM đã copy), chờ VM tạo ảnh
# 4. visual_ready - Có VISUAL + audio, chờ Edit
# 5. editing     - Đang edit (từ progress.json)
# 6. done        - Hoàn thành

class QueueTracker:
    """Track codes through the pipeline to detect smart states."""

    def __init__(self):
        self.tracker_file = QUEUE_TRACKER_FILE
        self.data = self.load()

    def load(self):
        """Load tracker data."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"sent_to_vm": {}}  # code -> timestamp

    def save(self):
        """Save tracker data."""
        try:
            with open(self.tracker_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except:
            pass

    def mark_sent_to_vm(self, code):
        """Mark a code as sent to VM (voice deleted after SRT)."""
        self.data["sent_to_vm"][code] = datetime.now().isoformat()
        self.save()

    def is_sent_to_vm(self, code):
        """Check if code was sent to VM."""
        return code in self.data.get("sent_to_vm", {})

    def clear_code(self, code):
        """Remove code from tracker (when VISUAL received or done)."""
        if code in self.data.get("sent_to_vm", {}):
            del self.data["sent_to_vm"][code]
            self.save()

    def get_all_codes(self):
        """Get all tracked codes and their states."""
        codes = {}

        # 1. Scan DONE folder first (completed)
        if DONE_DIR.exists():
            for folder in DONE_DIR.iterdir():
                if folder.is_dir() and any(folder.glob("*.mp4")):
                    code = folder.name
                    mp4_files = list(folder.glob("*.mp4"))
                    codes[code] = {
                        "status": "done",
                        "status_text": "Hoàn thành ✓",
                        "icon": "✅",
                        "color": COLORS["success"],
                        "mtime": mp4_files[0].stat().st_mtime if mp4_files else 0,
                        "path": mp4_files[0] if mp4_files else None
                    }
                    # Clear from tracker if done
                    self.clear_code(code)

        # 2. Scan VISUAL folder (ready for edit)
        if VISUAL_DIR.exists():
            thumb_dir = TOOL_DIR / "thumb"
            nv_dir = thumb_dir / "nv"
            thumbnail_dir = thumb_dir / "thumbnails"

            for folder in VISUAL_DIR.iterdir():
                if folder.is_dir():
                    code = folder.name
                    if code in codes:  # Already done
                        continue

                    has_audio = any(folder.glob("*.mp3")) or any(folder.glob("*.wav"))

                    # Check for images in both root folder and img\ subfolder (VM structure)
                    img_subfolder = folder / "img"
                    has_images = (
                        any(folder.glob("*.jpg")) or any(folder.glob("*.png")) or
                        (img_subfolder.exists() and (
                            any(img_subfolder.glob("*.jpg")) or
                            any(img_subfolder.glob("*.png")) or
                            any(img_subfolder.glob("*.mp4"))  # Videos count as media too
                        ))
                    )

                    if has_audio and has_images:
                        # Check for NV and thumbnail output
                        has_nv = (nv_dir / f"{code}.png").exists()
                        has_thumb = any(thumbnail_dir.glob(f"{code}.*"))

                        # If VISUAL exists with images, NV/Thumb CAN be generated
                        # (scripts will use VISUAL images as source)
                        will_generate = []
                        if not has_nv:
                            will_generate.append("NV")
                        if not has_thumb:
                            will_generate.append("Thumb")

                        if will_generate:
                            # Will be generated when Auto runs
                            codes[code] = {
                                "status": "visual_ready",
                                "status_text": f"Sẽ tạo {', '.join(will_generate)}",
                                "icon": "🔄",
                                "color": COLORS["accent_orange"]
                            }
                        else:
                            # All ready
                            codes[code] = {
                                "status": "visual_ready",
                                "status_text": "Sẵn sàng Edit",
                                "icon": "🎬",
                                "color": COLORS["accent_blue"]
                            }
                        # Clear from tracker since VISUAL received
                        self.clear_code(code)

        # 3. Scan voice folders
        if VOICE_DIR.exists():
            for subfolder in VOICE_DIR.iterdir():
                if subfolder.is_dir():
                    for ext in ['.mp3', '.wav', '.m4a']:
                        for audio_file in subfolder.glob(f"*{ext}"):
                            code = audio_file.stem
                            if code in codes:  # Already has VISUAL or done
                                continue

                            # Check if has SRT file (SRT done, waiting for VM to copy)
                            srt_file = audio_file.with_suffix('.srt')
                            if srt_file.exists():
                                codes[code] = {
                                    "status": "srt_done",
                                    "status_text": "Chờ VM copy",
                                    "icon": "📤",
                                    "color": COLORS["accent_orange"]
                                }
                                # Mark as sent to VM for tracking
                                self.mark_sent_to_vm(code)
                            else:
                                codes[code] = {
                                    "status": "voice",
                                    "status_text": "Chờ SRT",
                                    "icon": "🎤",
                                    "color": COLORS["accent_purple"]
                                }

        # 4. Check codes sent to VM but voice deleted (waiting for VM to create images)
        for code in list(self.data.get("sent_to_vm", {}).keys()):
            if code not in codes:
                # Voice was deleted, no VISUAL yet = waiting for VM
                codes[code] = {
                    "status": "waiting_vm",
                    "status_text": "Chờ VM tạo ảnh",
                    "icon": "🖼️",
                    "color": COLORS["warning"]
                }

        return codes


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================

class ModernButton(tk.Canvas):
    """Modern rounded button with hover effect."""

    def __init__(self, parent, text, command, color, width=120, height=40, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS["bg_card"], highlightthickness=0, **kwargs)

        self.command = command
        self.color = color
        self.hover_color = self._lighten_color(color)
        self.text = text
        self.width = width
        self.height = height
        self.disabled = False

        self._draw(color)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _lighten_color(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = min(255, r + 30)
        g = min(255, g + 30)
        b = min(255, b + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw(self, fill_color):
        self.delete("all")
        r = 8

        self.create_arc(0, 0, r*2, r*2, start=90, extent=90, fill=fill_color, outline="")
        self.create_arc(self.width-r*2, 0, self.width, r*2, start=0, extent=90, fill=fill_color, outline="")
        self.create_arc(0, self.height-r*2, r*2, self.height, start=180, extent=90, fill=fill_color, outline="")
        self.create_arc(self.width-r*2, self.height-r*2, self.width, self.height, start=270, extent=90, fill=fill_color, outline="")
        self.create_rectangle(r, 0, self.width-r, self.height, fill=fill_color, outline="")
        self.create_rectangle(0, r, self.width, self.height-r, fill=fill_color, outline="")

        text_color = "#ffffff" if not self.disabled else "#666666"
        self.create_text(self.width//2, self.height//2, text=self.text,
                        fill=text_color, font=("Segoe UI", 10, "bold"))

    def _on_enter(self, e):
        if not self.disabled:
            self._draw(self.hover_color)
            self.config(cursor="hand2")

    def _on_leave(self, e):
        if not self.disabled:
            self._draw(self.color)
            self.config(cursor="")

    def _on_click(self, e):
        if not self.disabled and self.command:
            self.command()

    def set_disabled(self, disabled):
        self.disabled = disabled
        self._draw("#4a4a4a" if disabled else self.color)




# ============================================================================
# MAIN APPLICATION
# ============================================================================

class VE3ToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VE3 Tool")
        self.root.geometry("900x700")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(True, True)
        self.root.minsize(700, 550)

        # Process handles
        self.srt_process = None
        self.thumb_process = None
        self.edit_process = None
        self.srt_running = False
        self.thumb_running = False
        self.edit_running = False
        self.auto_mode = False  # Auto mode runs all continuously
        self.parallel_var = tk.StringVar(value="auto")  # Default: auto-detect based on hardware

        # Smart queue tracker
        self.queue_tracker = QueueTracker()

        # Track failed codes from edit output
        self.failed_codes = {}  # code -> fail_count

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.create_ui()
        self.refresh_stats()
        self.update_progress()

    def on_closing(self):
        """Handle window close - kill all processes."""
        for proc in [self.srt_process, self.thumb_process, self.edit_process]:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except:
                    try:
                        proc.kill()
                    except:
                        pass

        self.root.destroy()

    def create_ui(self):
        self.current_tab = "edit_queue"

        main = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        self.create_header(main)
        self.create_process_bar(main)
        self.create_pipeline_panel(main)
        self.create_queue_tabs(main)
        self.create_log_area(main)

    def create_header(self, parent):
        """Header with Auto button, Parallel selector, Settings."""
        header = tk.Frame(parent, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 10))

        tk.Label(header, text="VE3 Tool", font=("Segoe UI", 16, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT)

        # Settings
        settings_btn = tk.Button(header, text="Cai dat", font=("Segoe UI", 9),
                                command=self.open_subtitle_settings,
                                bg=COLORS["bg_card"], fg=COLORS["text"],
                                relief=tk.FLAT, cursor="hand2", padx=8, pady=3)
        settings_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Update (git pull)
        update_btn = tk.Button(header, text="UPDATE", font=("Segoe UI", 9, "bold"),
                              command=self.update_from_github,
                              bg=COLORS["accent_blue"], fg="#ffffff",
                              relief=tk.FLAT, cursor="hand2", padx=8, pady=3)
        update_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Parallel selector
        par_frame = tk.Frame(header, bg=COLORS["bg_dark"])
        par_frame.pack(side=tk.RIGHT, padx=(0, 8))
        tk.Label(par_frame, text="Parallel:", font=("Segoe UI", 9),
                bg=COLORS["bg_dark"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
        ttk.Combobox(par_frame, textvariable=self.parallel_var,
                     values=["auto", "1", "2", "3", "4", "6", "8"],
                     width=5, state="readonly").pack(side=tk.LEFT, padx=(3, 0))

        # Auto button
        self.auto_btn = ModernButton(header, "Chay Auto", self.toggle_auto_mode,
                                    COLORS["accent_green"], width=100, height=32)
        self.auto_btn.pack(side=tk.RIGHT, padx=(0, 8))

    # ================================================================
    # PROCESS BAR - compact status indicators + controls
    # ================================================================

    def create_process_bar(self, parent):
        """Compact process status bar with indicators and start/stop."""
        bar = tk.Frame(parent, bg=COLORS["bg_card"],
                       highlightbackground=COLORS["border"], highlightthickness=1)
        bar.pack(fill=tk.X, pady=(0, 8))
        inner = tk.Frame(bar, bg=COLORS["bg_card"])
        inner.pack(fill=tk.X, padx=10, pady=5)

        self.srt_proc = self._make_proc_indicator(inner, "SRT", COLORS["accent_purple"],
                                                   self.start_srt, self.stop_srt)
        tk.Label(inner, text="│", bg=COLORS["bg_card"], fg=COLORS["border"],
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=8)
        self.thumb_proc = self._make_proc_indicator(inner, "Thumb/NV", COLORS["accent_orange"],
                                                     self.start_thumb, self.stop_thumb)
        tk.Label(inner, text="│", bg=COLORS["bg_card"], fg=COLORS["border"],
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=8)
        self.edit_proc = self._make_proc_indicator(inner, "Edit", COLORS["accent_blue"],
                                                    self.start_edit, self.stop_edit)

    def _make_proc_indicator(self, parent, name, color, on_start, on_stop):
        """Create one process indicator: ● Name info [▶][■]"""
        f = tk.Frame(parent, bg=COLORS["bg_card"])
        f.pack(side=tk.LEFT)

        dot = tk.Label(f, text="●", font=("Segoe UI", 11),
                       bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        dot.pack(side=tk.LEFT)

        lbl = tk.Label(f, text=name, font=("Segoe UI", 9, "bold"),
                       bg=COLORS["bg_card"], fg=color)
        lbl.pack(side=tk.LEFT, padx=(3, 0))

        info = tk.Label(f, text="dừng", font=("Segoe UI", 8),
                        bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        info.pack(side=tk.LEFT, padx=(5, 0))

        start_btn = ModernButton(f, "▶", on_start, color, width=28, height=22)
        start_btn.pack(side=tk.LEFT, padx=(5, 0))

        stop_btn = ModernButton(f, "■", on_stop, COLORS["accent_red"], width=28, height=22)
        stop_btn.pack(side=tk.LEFT, padx=(2, 0))
        stop_btn.set_disabled(True)

        return {
            "dot": dot, "info": info, "color": color,
            "start_btn": start_btn, "stop_btn": stop_btn,
        }

    def _set_proc_running(self, proc, running):
        """Update process indicator visual state."""
        if running:
            proc["dot"].config(fg=COLORS["success"])
            proc["info"].config(text="chạy", fg=COLORS["success"])
            proc["start_btn"].set_disabled(True)
            proc["stop_btn"].set_disabled(False)
        else:
            proc["dot"].config(fg=COLORS["text_dim"])
            proc["info"].config(text="dừng", fg=COLORS["text_dim"])
            proc["start_btn"].set_disabled(False)
            proc["stop_btn"].set_disabled(True)

    def _set_proc_info(self, proc, text):
        """Update the info text of a process indicator."""
        proc["info"].config(text=text, fg=COLORS["text"])

    # ================================================================
    # PIPELINE PANEL - per-code full pipeline status
    # ================================================================

    def create_pipeline_panel(self, parent):
        """Scrollable panel showing each active code's pipeline status."""
        header = tk.Frame(parent, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 3))
        self._pipeline_header = tk.Label(header, text="Đang xử lý",
                                          font=("Segoe UI", 11, "bold"),
                                          bg=COLORS["bg_dark"], fg=COLORS["text"])
        self._pipeline_header.pack(side=tk.LEFT)

        container = tk.Frame(parent, bg=COLORS["border"])
        container.pack(fill=tk.X, pady=(0, 8))

        canvas = tk.Canvas(container, bg=COLORS["bg_card"], highlightthickness=0, height=140)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        self._pipeline_frame = tk.Frame(canvas, bg=COLORS["bg_card"])
        self._pipeline_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._pipeline_frame, anchor="nw", tags="pf")

        def on_resize(event):
            canvas.itemconfig("pf", width=event.width)
        canvas.bind("<Configure>", on_resize)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._pipeline_canvas = canvas
        self._pipeline_widgets = {}
        self._pipeline_codes = []

        tk.Label(self._pipeline_frame, text="Chưa có mã nào đang xử lý",
                 font=("Segoe UI", 10), bg=COLORS["bg_card"],
                 fg=COLORS["text_dim"], pady=15).pack()

    def _build_pipeline_row(self, parent, code):
        """Build widgets for one code's pipeline row."""
        row = tk.Frame(parent, bg=COLORS["bg_card"])
        row.pack(fill=tk.X, padx=10, pady=4)

        # Line 1: code name + time
        line1 = tk.Frame(row, bg=COLORS["bg_card"])
        line1.pack(fill=tk.X)

        name_lbl = tk.Label(line1, text=code, font=("Consolas", 11, "bold"),
                            bg=COLORS["bg_card"], fg=COLORS["text"])
        name_lbl.pack(side=tk.LEFT)

        time_lbl = tk.Label(line1, text="", font=("Consolas", 9),
                            bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        time_lbl.pack(side=tk.RIGHT)

        # Line 2: pipeline badges → edit step + percent
        line2 = tk.Frame(row, bg=COLORS["bg_card"])
        line2.pack(fill=tk.X, pady=(2, 0))

        srt_badge = tk.Label(line2, text="SRT ✓", font=("Segoe UI", 8, "bold"),
                             bg=COLORS["bg_card"], fg=COLORS["success"])
        srt_badge.pack(side=tk.LEFT)

        thumb_badge = tk.Label(line2, text="Thumb ...", font=("Segoe UI", 8, "bold"),
                               bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        thumb_badge.pack(side=tk.LEFT, padx=(6, 0))

        nv_badge = tk.Label(line2, text="NV ...", font=("Segoe UI", 8, "bold"),
                            bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        nv_badge.pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(line2, text="→", font=("Segoe UI", 9),
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(6, 6))

        step_lbl = tk.Label(line2, text="", font=("Segoe UI", 9),
                            bg=COLORS["bg_card"], fg=COLORS["accent_blue"])
        step_lbl.pack(side=tk.LEFT)

        pct_lbl = tk.Label(line2, text="", font=("Consolas", 10, "bold"),
                           bg=COLORS["bg_card"], fg=COLORS["text"])
        pct_lbl.pack(side=tk.RIGHT)

        # Line 3: progress bar
        bar = tk.Canvas(row, height=6, bg=COLORS["bg_dark"], highlightthickness=0)
        bar.pack(fill=tk.X, pady=(3, 0))

        return {
            "frame": row, "name": name_lbl, "time": time_lbl,
            "srt_badge": srt_badge, "thumb_badge": thumb_badge, "nv_badge": nv_badge,
            "step": step_lbl, "percent": pct_lbl, "progress_bar": bar,
        }

    def _update_pipeline(self, videos):
        """Update pipeline panel with current video data."""
        codes = sorted(videos.keys())

        # Rebuild if codes changed
        if codes != self._pipeline_codes:
            self._pipeline_codes = codes
            for w in self._pipeline_frame.winfo_children():
                w.destroy()
            self._pipeline_widgets = {}

            if not codes:
                tk.Label(self._pipeline_frame, text="Chưa có mã nào đang xử lý",
                         font=("Segoe UI", 10), bg=COLORS["bg_card"],
                         fg=COLORS["text_dim"], pady=15).pack()
                self._pipeline_header.config(text="Đang xử lý")
                return

            for i, code in enumerate(codes):
                if i > 0:
                    tk.Frame(self._pipeline_frame, bg=COLORS["border"],
                             height=1).pack(fill=tk.X, padx=5, pady=1)
                self._pipeline_widgets[code] = self._build_pipeline_row(self._pipeline_frame, code)

        self._pipeline_header.config(
            text=f"Đang xử lý ({len(codes)} mã)" if codes else "Đang xử lý")

        # Update each row
        thumb_dir = TOOL_DIR / "thumb" / "thumbnails"
        nv_dir = TOOL_DIR / "thumb" / "nv"

        for code in codes:
            if code not in self._pipeline_widgets:
                continue
            w = self._pipeline_widgets[code]
            vid = videos.get(code, {})

            # Badges
            w["srt_badge"].config(text="SRT ✓", fg=COLORS["success"])

            has_thumb = any(thumb_dir.glob(f"{code}.*")) if thumb_dir.exists() else False
            w["thumb_badge"].config(
                text="Thumb ✓" if has_thumb else "Thumb ...",
                fg=COLORS["success"] if has_thumb else COLORS["text_dim"])

            has_nv = (nv_dir / f"{code}.png").exists() if nv_dir.exists() else False
            w["nv_badge"].config(
                text="NV ✓" if has_nv else "NV ...",
                fg=COLORS["success"] if has_nv else COLORS["text_dim"])

            # Edit status
            step = vid.get("step", "")
            percent = vid.get("percent", 0)
            clip_info = ""
            if vid.get("clip_total", 0) > 0 and "clip" in step.lower():
                clip_info = f" ({vid.get('clip_current', 0)}/{vid['clip_total']})"

            fail_count = self.failed_codes.get(code, 0)
            if fail_count >= 3:
                w["step"].config(text=f"LỖI - đã skip ({fail_count} lần)", fg=COLORS["error"])
                w["percent"].config(text="✗", fg=COLORS["error"])
            elif fail_count > 0:
                w["step"].config(text=f"Edit: {step}{clip_info} (lỗi {fail_count}/3)",
                                 fg=COLORS["warning"])
                w["percent"].config(text=f"{percent}%", fg=COLORS["warning"])
            elif step.lower() == "done":
                w["step"].config(text="Done ✓", fg=COLORS["success"])
                w["percent"].config(text="100%", fg=COLORS["success"])
            else:
                w["step"].config(text=f"Edit: {step}{clip_info}", fg=COLORS["accent_blue"])
                w["percent"].config(text=f"{percent}%", fg=COLORS["text"])

            # Time (live elapsed)
            elapsed_seconds = vid.get("elapsed_seconds", 0)
            started_at = vid.get("started_at")
            if started_at:
                try:
                    from datetime import datetime as dt
                    start_time = dt.fromisoformat(started_at)
                    elapsed_seconds = int((dt.now() - start_time).total_seconds())
                except:
                    pass

            eta_seconds = vid.get("eta_seconds")
            time_parts = [self._format_time(elapsed_seconds)]
            if eta_seconds and eta_seconds > 0:
                time_parts.append(f"ETA {self._format_time(eta_seconds)}")
            w["time"].config(text="  ".join(time_parts))

            # Progress bar
            pbar = w["progress_bar"]
            pbar.delete("all")
            bar_w = pbar.winfo_width()
            if bar_w > 1 and percent > 0:
                fill_w = int(bar_w * percent / 100)
                bar_color = COLORS["error"] if fail_count >= 3 else COLORS["accent_green"]
                pbar.create_rectangle(0, 0, fill_w, 6, fill=bar_color, outline="")

    # ================================================================
    # QUEUE TABS - Cho Edit / Loi / Hoan thanh
    # ================================================================

    def create_queue_tabs(self, parent):
        """Tabbed queue: Cho Edit | Loi | Hoan thanh + VM count."""
        container = tk.Frame(parent, bg=COLORS["bg_dark"])
        container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Tab bar
        tab_bar = tk.Frame(container, bg=COLORS["bg_dark"])
        tab_bar.pack(fill=tk.X, pady=(0, 5))

        self.tab_buttons = {}
        tabs = [
            ("edit_queue", "Cho Edit", COLORS["accent_blue"]),
            ("errors", "Loi", COLORS["error"]),
            ("done", "Hoan thanh", COLORS["success"]),
        ]
        for tab_id, label, color in tabs:
            btn = tk.Label(tab_bar, text=f" {label}: 0 ", font=("Segoe UI", 10, "bold"),
                          bg=COLORS["bg_card"], fg=color, cursor="hand2",
                          padx=10, pady=4)
            btn.pack(side=tk.LEFT, padx=(0, 3))
            btn.bind("<Button-1>", lambda e, t=tab_id: self._switch_tab(t))
            self.tab_buttons[tab_id] = btn

        # VM count (compact, right side)
        self.vm_count_label = tk.Label(tab_bar, text="", font=("Segoe UI", 9),
                                       bg=COLORS["bg_dark"], fg=COLORS["text_dim"])
        self.vm_count_label.pack(side=tk.RIGHT)

        # Tab content area
        content_frame = tk.Frame(container, bg=COLORS["border"])
        content_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(content_frame, bg=COLORS["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)

        self.tab_content_frame = tk.Frame(canvas, bg=COLORS["bg_card"])
        self.tab_content_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.tab_content_frame, anchor="nw",
                            tags="content_window")

        # Make content frame fill canvas width
        def on_canvas_resize(event):
            canvas.itemconfig("content_window", width=event.width)
        canvas.bind("<Configure>", on_canvas_resize)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        self._tab_canvas = canvas
        self._switch_tab("edit_queue")

    def _switch_tab(self, tab_id):
        """Switch active tab."""
        self.current_tab = tab_id
        # Update tab button styling
        for tid, btn in self.tab_buttons.items():
            if tid == tab_id:
                btn.config(bg=COLORS["accent_blue"] if tid == "edit_queue"
                          else COLORS["error"] if tid == "errors"
                          else COLORS["success"],
                          fg="#ffffff")
            else:
                btn.config(bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self._render_tab_content()

    def _render_tab_content(self):
        """Populate the active tab with items."""
        for w in self.tab_content_frame.winfo_children():
            w.destroy()

        all_codes = self.queue_tracker.get_all_codes()

        if self.current_tab == "edit_queue":
            items = [(c, d) for c, d in all_codes.items()
                     if d["status"] in ("visual_ready", "editing")]
            items.sort(key=lambda x: (0 if x[1]["status"] == "editing" else 1, x[0]))

            if not items:
                tk.Label(self.tab_content_frame, text="Khong co ma nao cho edit",
                        font=("Segoe UI", 10), bg=COLORS["bg_card"],
                        fg=COLORS["text_dim"], pady=20).pack()
            else:
                for code, data in items:
                    # Override with failed status if applicable
                    fail_count = self.failed_codes.get(code, 0)
                    if fail_count >= 3:
                        data = dict(data, icon="X", status_text="LOI - Excel/data hong",
                                   color=COLORS["error"])
                    elif fail_count > 0:
                        data = dict(data, icon="!", status_text=f"Loi (thu lai {fail_count}/3)",
                                   color=COLORS["warning"])
                    self._render_queue_item(code, data)

        elif self.current_tab == "errors":
            error_items = [(c, d) for c, d in all_codes.items()
                          if c in self.failed_codes]
            if not error_items:
                tk.Label(self.tab_content_frame, text="Khong co loi",
                        font=("Segoe UI", 10), bg=COLORS["bg_card"],
                        fg=COLORS["success"], pady=20).pack()
            else:
                for code, data in error_items:
                    fail_count = self.failed_codes.get(code, 0)
                    if fail_count >= 3:
                        data = dict(data, icon="X", status_text="LOI - Excel/data hong, da skip",
                                   color=COLORS["error"])
                    else:
                        data = dict(data, icon="!", status_text=f"Loi (thu lai {fail_count}/3)",
                                   color=COLORS["warning"])
                    self._render_queue_item(code, data)

        elif self.current_tab == "done":
            done_items = [(c, d) for c, d in all_codes.items() if d["status"] == "done"]
            done_items.sort(key=lambda x: -x[1].get("mtime", 0))
            if not done_items:
                tk.Label(self.tab_content_frame, text="Chua co ma nao hoan thanh",
                        font=("Segoe UI", 10), bg=COLORS["bg_card"],
                        fg=COLORS["text_dim"], pady=20).pack()
            else:
                timing_dict = self._load_timing_dict()
                for code, data in done_items[:30]:
                    if code in timing_dict:
                        steps = timing_dict[code].get("steps_s", {})
                        total_s = steps.get("total", 0)
                        sub_s = steps.get("subtitle_burn", 0)
                        total_min = round(total_s / 60)
                        sub_min = round(sub_s / 60)
                        data = dict(data)
                        data["status_text"] = f"✓ {total_min}p (sub {sub_min}p)"
                    self._render_queue_item(code, data, clickable=True)

    def _render_queue_item(self, code, data, clickable=False):
        """Render one item in the queue tab."""
        row = tk.Frame(self.tab_content_frame, bg=COLORS["bg_card"])
        row.pack(fill=tk.X, padx=10, pady=1)

        is_error = data.get("color") == COLORS["error"]
        is_editing = data.get("status") == "editing"
        bg = COLORS["accent_blue"] if is_editing else COLORS["bg_card"]
        fg = "#ffffff" if is_editing else COLORS["text"]

        icon = data.get("icon", "●")
        icon_color = COLORS["error"] if is_error else data.get("color", COLORS["text_dim"])
        tk.Label(row, text=icon, font=("Segoe UI", 10), bg=bg,
                fg=icon_color).pack(side=tk.LEFT, padx=(0, 5))

        font = ("Segoe UI", 10, "bold") if is_editing else ("Segoe UI", 10)
        code_lbl = tk.Label(row, text=code, font=font, bg=bg, fg=fg)
        code_lbl.pack(side=tk.LEFT)

        status_text = data.get("status_text", "")
        if status_text:
            st_color = COLORS["error"] if is_error else COLORS["text_dim"]
            tk.Label(row, text=status_text, font=("Segoe UI", 9),
                    bg=bg, fg=st_color).pack(side=tk.RIGHT, padx=(0, 5))

        if clickable and data.get("path"):
            row.config(cursor="hand2")
            code_lbl.config(cursor="hand2")
            path = data["path"]
            folder = path.parent if path else None
            if folder:
                for w in [row, code_lbl]:
                    w.bind("<Button-1>", lambda e, p=folder: os.startfile(str(p)))
                    w.bind("<Enter>", lambda e, f=row: f.configure(bg=COLORS["bg_card_hover"]))
                    w.bind("<Leave>", lambda e, f=row: f.configure(bg=COLORS["bg_card"]))

    def refresh_code_list(self):
        """Refresh queue tabs and counts."""
        all_codes = self.queue_tracker.get_all_codes()

        # Count by category
        edit_count = sum(1 for d in all_codes.values() if d["status"] in ("visual_ready", "editing"))
        error_count = len(self.failed_codes)
        done_count = sum(1 for d in all_codes.values() if d["status"] == "done")
        vm_count = sum(1 for d in all_codes.values()
                      if d["status"] in ("voice", "srt_done", "waiting_vm"))

        # Update tab buttons with counts
        self.tab_buttons["edit_queue"].config(text=f" Cho Edit: {edit_count} ")
        self.tab_buttons["errors"].config(text=f" Loi: {error_count} ")
        self.tab_buttons["done"].config(text=f" Hoan thanh: {done_count} ")

        # Show/hide error tab highlight
        if error_count > 0:
            self.tab_buttons["errors"].config(fg=COLORS["error"])
        else:
            self.tab_buttons["errors"].config(fg=COLORS["text_dim"])

        # VM count
        if vm_count > 0:
            self.vm_count_label.config(text=f"{vm_count} ma cho VM")
        else:
            self.vm_count_label.config(text="")

        # Re-render current tab
        self._render_tab_content()

    def refresh_all(self):
        self.refresh_code_list()
        self.log("Da lam moi", "info")

    def play_video(self, video_path):
        try:
            os.startfile(str(video_path))
            self.log(f"Playing: {video_path.name}", "info")
        except Exception as e:
            self.log(f"Cannot play video: {e}", "error")

    # show_progress/hide_progress removed - handled by pipeline panel


    def create_log_area(self, parent):
        log_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        log_frame.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_frame, bg=COLORS["bg_dark"])
        log_header.pack(fill=tk.X, pady=(0, 5))

        tk.Label(log_header, text="📋 Activity Log", font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT)

        clear_btn = tk.Label(log_header, text="Clear", font=("Segoe UI", 9),
                            bg=COLORS["bg_dark"], fg=COLORS["accent_blue"], cursor="hand2")
        clear_btn.pack(side=tk.RIGHT)
        clear_btn.bind("<Button-1>", lambda e: self.clear_log())

        log_container = tk.Frame(log_frame, bg=COLORS["border"])
        log_container.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_container, height=10,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            font=("Consolas", 9),
            relief=tk.FLAT, wrap=tk.WORD,
            insertbackground=COLORS["text"],
            selectbackground=COLORS["accent_blue"],
            padx=10, pady=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self.log_text.tag_configure("info", foreground=COLORS["text"])
        self.log_text.tag_configure("success", foreground=COLORS["success"])
        self.log_text.tag_configure("warning", foreground=COLORS["warning"])
        self.log_text.tag_configure("error", foreground=COLORS["error"])
        self.log_text.tag_configure("time", foreground=COLORS["text_dim"])

        self.log("VE3 Tool GUI initialized", "success")
        self.log(f"VISUAL: {VISUAL_DIR}")
        self.log(f"DONE: {DONE_DIR}")

    def log(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] ", "time")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared", "info")

    def update_progress(self):
        """Read progress file and update pipeline panel."""
        try:
            if PROGRESS_FILE.exists() and self.edit_running:
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    progress = json.load(f)

                videos = progress.get("videos", {})

                # Support old flat format (single video)
                if "code" in progress and "step" in progress:
                    videos = {progress["code"]: progress}

                # Update pipeline panel with per-code data
                self._update_pipeline(videos)

                # Update process bar info with count
                if videos:
                    self._set_proc_info(self.edit_proc, f"chạy ({len(videos)} mã)")

                # Update window title from most advanced video
                best = None
                for vid_code, vid in videos.items():
                    step = vid.get("step", "")
                    if not step:
                        continue
                    if best is None or (vid.get("percent", 0) > best.get("percent", 0)
                                        and step.lower() != "done"):
                        best = vid

                if best:
                    code = best.get("code", "")
                    percent = best.get("percent", 0)
                    eta_seconds = best.get("eta_seconds")
                    if eta_seconds and eta_seconds > 0:
                        self.root.title(f"VE3 Tool - {code} {percent}% (ETA {self._format_time(eta_seconds)})")
                    else:
                        self.root.title(f"VE3 Tool - {percent}%")

                # Refresh queue tabs periodically
                if not hasattr(self, '_last_refresh_time'):
                    self._last_refresh_time = 0
                now = time.time()
                if (now - self._last_refresh_time) > 5:
                    self._last_refresh_time = now
                    self.refresh_code_list()
            else:
                if not self.edit_running:
                    self._update_pipeline({})
                    self.root.title("VE3 Tool")
        except Exception as e:
            print(f"[GUI] update_progress error: {e}")

        self.root.after(1000, self.update_progress)

    def _format_time(self, seconds, always_hours=False):
        """Format seconds to HH:MM:SS."""
        if seconds is None or seconds < 0:
            return "--:--:--" if always_hours else "--:--"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if always_hours or hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


    def refresh_stats(self):
        """Refresh stats và code list."""
        self.refresh_code_list()
        self.root.after(10000, self.refresh_stats)

    def _load_timing_dict(self):
        """Load timing_log.json as {code: latest_entry}."""
        timing_file = TOOL_DIR / "timing_log.json"
        if not timing_file.exists():
            return {}
        try:
            with open(timing_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
            result = {}
            for entry in entries:
                code = entry.get("code", "")
                if code:
                    result[code] = entry  # last entry wins
            return result
        except Exception:
            return {}

    def toggle_auto_mode(self):
        """Toggle auto mode - runs SRT, Thumb/NV, and Edit continuously."""
        if self.auto_mode:
            # Stop auto mode
            self.auto_mode = False
            self.auto_btn.text = "▶ Chạy Auto"
            self.auto_btn.color = COLORS["accent_green"]
            self.auto_btn._draw(COLORS["accent_green"])
            self.log("Auto mode stopped", "warning")
            self.root.title("VE3 Tool")
            # Stop all processes
            self.stop_srt()
            self.stop_thumb()
            self.stop_edit()
        else:
            # Start auto mode
            self.auto_mode = True
            self.auto_btn.text = "■ Stop"
            self.auto_btn.color = COLORS["accent_red"]
            self.auto_btn._draw(COLORS["accent_red"])
            self.root.title("VE3 Tool - Auto Running")
            self.log("Auto mode started - SRT → Thumb/NV → Edit", "success")
            # Start all processes
            if not self.srt_running:
                self.start_srt()
            if not self.thumb_running:
                self.start_thumb()
            if not self.edit_running:
                self.start_edit()


    def start_srt(self):
        if self.srt_running:
            return

        self.log("Starting SRT Generator...", "info")
        self.srt_running = True
        self._set_proc_running(self.srt_proc, True)

        def run():
            try:
                # CREATE_NO_WINDOW flag to hide console
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                # Set UTF-8 encoding for subprocess
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                self.srt_process = subprocess.Popen(
                    [sys.executable, str(TOOL_DIR / "run_srt.py")],
                    cwd=str(TOOL_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    encoding='utf-8', errors='replace',
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    env=env
                )

                for line in self.srt_process.stdout:
                    line = line.strip()
                    if line:
                        level = "success" if "Done" in line or "OK" in line else "info"
                        if "Error" in line or "ERROR" in line:
                            level = "error"
                        # Parse code from SRT output to show in status
                        self.root.after(0, lambda l=line, lv=level: self._on_srt_line(l, lv))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"[SRT] Error: {e}", "error"))
            finally:
                self.root.after(0, self._on_srt_stopped)

        threading.Thread(target=run, daemon=True).start()

    def _on_srt_line(self, line, level):
        """Process SRT output line - log and update status with current code."""
        self.log(f"[SRT] {line}", level)
        # Try to detect code from line (e.g. "Processing KA1-0001" or "[KA1-0001]")
        import re
        m = re.search(r'(KA\d+-\d+)', line)
        if m:
            code = m.group(1)
            self._set_proc_info(self.srt_proc, f"> {code}")

    def stop_srt(self):
        if self.srt_process:
            self.log("Stopping SRT Generator...", "warning")
            self.srt_process.terminate()

    def _on_srt_stopped(self):
        self.srt_running = False
        self._set_proc_running(self.srt_proc, False)
        self.log("SRT Generator stopped", "warning")
        self.refresh_code_list()
        # Restart if auto mode is on
        if self.auto_mode:
            self.root.after(3000, self.start_srt)  # Wait 3s then restart

    def start_thumb(self):
        """Start thumbnail and NV generation."""
        if self.thumb_running:
            return

        self.log("Starting Thumb/NV Generator...", "info")
        self.thumb_running = True
        self._set_proc_running(self.thumb_proc, True)

        def run():
            try:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                # Set UTF-8 encoding for subprocess
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                self.thumb_process = subprocess.Popen(
                    [sys.executable, str(TOOL_DIR / "run_thumb.py")],
                    cwd=str(TOOL_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    encoding='utf-8', errors='replace',
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    env=env
                )

                for line in self.thumb_process.stdout:
                    line = line.strip()
                    if line:
                        level = "success" if "[OK]" in line or "Done" in line else "info"
                        if "[ERR]" in line or "Error" in line:
                            level = "error"
                        self.root.after(0, lambda l=line, lv=level: self._on_thumb_line(l, lv))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"[THUMB] Error: {e}", "error"))
            finally:
                self.root.after(0, self._on_thumb_stopped)

        threading.Thread(target=run, daemon=True).start()

    def stop_thumb(self):
        if self.thumb_process:
            self.log("Stopping Thumb/NV Generator...", "warning")
            self.thumb_process.terminate()

    def _on_thumb_line(self, line, level):
        """Process Thumb output line - log and update status with current code."""
        self.log(f"[THUMB] {line}", level)
        import re
        m = re.search(r'(KA\d+-\d+)', line)
        if m:
            code = m.group(1)
            self._set_proc_info(self.thumb_proc, f"> {code}")

    def _on_thumb_stopped(self):
        self.thumb_running = False
        self._set_proc_running(self.thumb_proc, False)
        self.log("Thumb/NV Generator stopped", "warning")
        self.refresh_code_list()
        # Restart if auto mode is on
        if self.auto_mode:
            self.root.after(5000, self.start_thumb)  # Wait 5s then restart

    def start_edit(self):
        if self.edit_running:
            return

        parallel = self.parallel_var.get()
        if parallel == "auto":
            self.log("Starting Video Editor (auto-detect hardware)...", "info")
        else:
            self.log(f"Starting Video Editor (parallel={parallel})...", "info")
        self.edit_running = True
        self._set_proc_running(self.edit_proc, True)

        def run():
            try:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                # Set UTF-8 encoding for subprocess
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                # Build command - don't pass --parallel if auto
                cmd = [sys.executable, str(TOOL_DIR / "run_edit.py")]
                if parallel != "auto":
                    cmd.extend(["--parallel", parallel])

                self.edit_process = subprocess.Popen(
                    cmd,
                    cwd=str(TOOL_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    encoding='utf-8', errors='replace',
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    env=env
                )

                for line in self.edit_process.stdout:
                    line = line.strip()
                    if line:
                        level = "success" if "DONE" in line or "OK" in line else "info"
                        if "Error" in line or "ERROR" in line or "FAIL" in line:
                            level = "error"
                        self.root.after(0, lambda l=line, lv=level: self._on_edit_line(l, lv))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"[EDIT] Error: {e}", "error"))
            finally:
                self.root.after(0, self._on_edit_stopped)

        threading.Thread(target=run, daemon=True).start()

    def stop_edit(self):
        if self.edit_process:
            self.log("Stopping Video Editor...", "warning")
            self.edit_process.terminate()

    def _on_edit_line(self, line, level):
        """Process Edit output line - log, track failures, update status."""
        self.log(f"[EDIT] {line}", level)
        import re
        # Detect FAILED: "KA2-0006: FAILED"
        m = re.search(r'(KA\d+-\d+):\s*FAILED', line)
        if m:
            code = m.group(1)
            self.failed_codes[code] = self.failed_codes.get(code, 0) + 1
            self.refresh_code_list()
        # Detect SUCCESS: "KA2-0007: SUCCESS"
        m = re.search(r'(KA\d+-\d+):\s*SUCCESS', line)
        if m:
            code = m.group(1)
            self.failed_codes.pop(code, None)
            self.refresh_code_list()
        # Detect SKIP: "SKIPPING from now on"
        m = re.search(r'(KA\d+-\d+):\s*FAILED.*SKIPPING', line)
        if m:
            code = m.group(1)
            self.failed_codes[code] = 99  # Permanent skip

    def _on_edit_stopped(self):
        self.edit_running = False
        self._set_proc_running(self.edit_proc, False)
        self.log("Video Editor stopped", "warning")
        self.refresh_code_list()
        # Restart if auto mode is on
        if self.auto_mode:
            self.root.after(3000, self.start_edit)  # Wait 3s then restart

    def update_from_github(self):
        """Pull latest code from GitHub."""
        self.log("Đang cập nhật từ GitHub...", "info")

        def do_update():
            try:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                result = subprocess.run(["git", "pull", "origin", "main"],
                                        cwd=str(TOOL_DIR), capture_output=True, text=True,
                                        startupinfo=startupinfo)

                if result.returncode == 0:
                    output = result.stdout.strip()
                    if "Already up to date" in output:
                        self.root.after(0, lambda: self.log("Đã là phiên bản mới nhất.", "info"))
                    else:
                        self.root.after(0, lambda: self.log(f"Cập nhật thành công! {output}", "success"))
                else:
                    self.root.after(0, lambda: self.log(f"Lỗi update: {result.stderr}", "error"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Update error: {e}", "error"))

        threading.Thread(target=do_update, daemon=True).start()

    def upload_github(self):
        self.log("Uploading to GitHub...", "info")

        def do_upload():
            try:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                subprocess.run(["git", "add", "-A"], cwd=str(TOOL_DIR),
                              capture_output=True, startupinfo=startupinfo)
                subprocess.run(["git", "commit", "-m", f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
                              cwd=str(TOOL_DIR), capture_output=True, startupinfo=startupinfo)
                result = subprocess.run(["git", "push", "origin", "main"],
                                        cwd=str(TOOL_DIR), capture_output=True, text=True,
                                        startupinfo=startupinfo)

                if result.returncode == 0:
                    self.root.after(0, lambda: self.log("Upload successful!", "success"))
                else:
                    self.root.after(0, lambda: self.log(f"Upload: {result.stderr or 'Done'}", "info"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Upload error: {e}", "error"))

        threading.Thread(target=do_upload, daemon=True).start()

    def open_subtitle_settings(self):
        """Open subtitle template settings dialog."""
        SubtitleTemplateDialog(self.root, self)

    # Video settings are now part of Template dialog


# ============================================================================
# CHANNEL SETTINGS DIALOG - FULL SETTINGS
# ============================================================================

class SubtitleTemplateDialog:
    """Full channel settings dialog with all options organized by section."""

    TEMPLATES_FILE = TOOL_DIR / "subtitle_templates.json"
    THUMB_OUTPUT_DIR = TOOL_DIR / "thumb" / "thumbnails"
    NV_OUTPUT_DIR = TOOL_DIR / "thumb" / "nv"
    FONTS_DIR = TOOL_DIR / "fonts"
    DONE_DIR = Path(r"D:\AUTO\done")

    # Options
    RESOLUTIONS = {"4K (3840x2160)": "4k", "2K (2560x1440)": "2k", "1080p (1920x1080)": "1080p"}
    TRANSITIONS = {"Ngẫu nhiên": "random", "Fade đen": "fade_black", "Mix": "mix"}
    KEN_BURNS = {"Nhẹ": "subtle", "Vừa": "medium", "Mạnh": "strong", "Tắt": "none"}
    NV_H_POSITIONS = {"Trái": "left", "Phải": "right"}
    NV_V_POSITIONS = {"Trên": "top", "Giữa": "middle", "Dưới": "bottom"}
    SUBTITLE_COLORS = {
        "Trắng": "&H00FFFFFF",
        "Vàng": "&H0000FFFF",
        "Xanh dương": "&H00FF0000",
        "Đỏ": "&H000000FF",
        "Xanh lá": "&H0000FF00",
    }
    # ASS alignment: 1-3=bottom, 4-6=middle, 7-9=top (left/center/right)
    SUBTITLE_POSITIONS = {"Dưới": 2, "Giữa": 5, "Trên": 8}

    DEFAULT_TEMPLATE = {
        # Video
        "output_resolution": "4k",
        "video_transition": "random",
        "ken_burns_intensity": "subtle",
        # NV Overlay
        "nv_overlay_enabled": True,
        "nv_overlay_position": "left",
        "nv_overlay_v_position": "middle",
        "nv_overlay_scale": 0.50,
        # Subtitle
        "font": "Bebas Neue",
        "size": 28,
        "color": "&H00FFFFFF",
        "outline": "&H00000000",
        "outline_size": 2,
        "margin_v": 25,
        "alignment": 2,  # 2=bottom, 5=middle, 8=top (ASS format)
        # Other
        "compose_mode": "quality",
    }

    def __init__(self, parent, app):
        self.app = app
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("⚙ Cài đặt Channel")
        self.dialog.configure(bg=COLORS["bg_dark"])
        self.dialog.geometry("700x780")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)

        self.templates = self.load_templates()
        self.current_channel = None
        self.preview_image = None
        self.video_preview_image = None
        self.available_fonts = self._scan_fonts()
        self.all_channels = self._scan_channels()

        self.create_ui()

        # Center dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")

    def _scan_fonts(self):
        """Scan fonts directory for available fonts."""
        fonts = []
        if self.FONTS_DIR.exists():
            for f in self.FONTS_DIR.glob("*.ttf"):
                # Clean font name from filename
                name = f.stem.replace("-", " ").replace("_", " ")
                # Remove common suffixes
                for suffix in ["Regular", "Bold", "Italic", "VariableFont", "wght", "opsz"]:
                    name = name.replace(suffix, "").strip()
                if name and name not in fonts:
                    fonts.append(name)
        return sorted(set(fonts)) if fonts else ["Bebas Neue", "Arial", "Roboto"]

    def _scan_channels(self):
        """Scan thumb folder for all available channels."""
        channels = set()

        # 1. From saved templates
        channels.update(self.templates.keys())

        # 2. From thumb scripts (pattern: *.NV_CHANNEL.py or *.THUMB_CHANNEL.py)
        thumb_dir = TOOL_DIR / "thumb"
        if thumb_dir.exists():
            import re
            for f in thumb_dir.glob("*.py"):
                # Match patterns like 4.NV_KA1-T5.py or 7.THUMB_KA2-T2.py
                match = re.search(r'(?:NV|THUMB)_([A-Z0-9]+-T\d+)', f.name, re.IGNORECASE)
                if match:
                    channels.add(match.group(1).upper())

        # Sort: group by channel prefix (KA1, KA2, etc.)
        return sorted(channels, key=lambda x: (x.split("-")[0], int(x.split("-T")[-1]) if "-T" in x else 0))

    def load_templates(self):
        if self.TEMPLATES_FILE.exists():
            try:
                with open(self.TEMPLATES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_templates(self):
        try:
            with open(self.TEMPLATES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.app.log(f"Error saving templates: {e}", "error")

    def create_ui(self):
        # Main scrollable frame
        main = tk.Frame(self.dialog, bg=COLORS["bg_dark"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # === HEADER: Channel + Save ===
        header = tk.Frame(main, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 15))

        tk.Label(header, text="📺 Channel:", font=("Segoe UI", 12, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT)

        # Use all_channels (scanned from thumb scripts + saved templates)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(header, textvariable=self.channel_var,
                                         values=self.all_channels, width=12, font=("Segoe UI", 11))
        self.channel_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_selected)
        self.channel_combo.bind("<Return>", self.on_channel_selected)

        # Save button
        save_btn = tk.Button(header, text="💾 Lưu", command=self.save_template,
                            bg=COLORS["accent_green"], fg="white", relief=tk.FLAT,
                            font=("Segoe UI", 11, "bold"), padx=20, pady=5, cursor="hand2")
        save_btn.pack(side=tk.RIGHT)

        # Delete button
        del_btn = tk.Button(header, text="🗑", command=self.delete_template,
                           bg=COLORS["accent_red"], fg="white", relief=tk.FLAT,
                           font=("Segoe UI", 11), padx=10, pady=5, cursor="hand2")
        del_btn.pack(side=tk.RIGHT, padx=(0, 10))

        # === PREVIEW SECTION (Thumbnail + Video) ===
        self._create_section(main, "🖼 Xem trước", self._create_preview_content)

        # === VIDEO SECTION ===
        self._create_section(main, "🎬 Video", self._create_video_content)

        # === NV OVERLAY SECTION ===
        self._create_section(main, "👤 NV Overlay", self._create_nv_content)

        # === SUBTITLE SECTION ===
        self._create_section(main, "💬 Phụ đề", self._create_subtitle_content)

    def _create_section(self, parent, title, content_func):
        """Create a collapsible section with title and content."""
        section = tk.Frame(parent, bg=COLORS["bg_card"], highlightbackground=COLORS["border"],
                          highlightthickness=1)
        section.pack(fill=tk.X, pady=(0, 12))

        # Section header
        header = tk.Frame(section, bg=COLORS["bg_card"])
        header.pack(fill=tk.X, padx=15, pady=(12, 8))

        tk.Label(header, text=title, font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_card"], fg=COLORS["accent_blue"]).pack(side=tk.LEFT)

        # Section content
        content = tk.Frame(section, bg=COLORS["bg_card"])
        content.pack(fill=tk.X, padx=15, pady=(0, 12))
        content_func(content)

    def _create_preview_content(self, parent):
        """Preview section - shows thumbnail and video frame side by side."""
        # Container for both previews
        preview_row = tk.Frame(parent, bg=COLORS["bg_card"])
        preview_row.pack(fill=tk.X, pady=5)

        # Left: Thumbnail preview
        thumb_frame = tk.Frame(preview_row, bg=COLORS["bg_dark"], width=300, height=170)
        thumb_frame.pack(side=tk.LEFT, padx=(0, 10))
        thumb_frame.pack_propagate(False)

        tk.Label(thumb_frame, text="Thumbnail", font=("Segoe UI", 9, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text_dim"]).pack(anchor=tk.NW, padx=5, pady=2)
        self.preview_label = tk.Label(thumb_frame, bg=COLORS["bg_dark"],
                                     text="Chọn channel",
                                     fg=COLORS["text_dim"], font=("Segoe UI", 9))
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Right: Video frame preview
        video_frame = tk.Frame(preview_row, bg=COLORS["bg_dark"], width=300, height=170)
        video_frame.pack(side=tk.LEFT)
        video_frame.pack_propagate(False)

        tk.Label(video_frame, text="Video (NV + Sub)", font=("Segoe UI", 9, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text_dim"]).pack(anchor=tk.NW, padx=5, pady=2)
        self.video_preview_label = tk.Label(video_frame, bg=COLORS["bg_dark"],
                                           text="Chọn channel",
                                           fg=COLORS["text_dim"], font=("Segoe UI", 9))
        self.video_preview_label.pack(fill=tk.BOTH, expand=True)

    def _create_video_content(self, parent):
        """Video settings section."""
        # Row 1: Resolution + Transition
        row1 = tk.Frame(parent, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, pady=5)

        # Resolution
        tk.Label(row1, text="Độ phân giải:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.resolution_var = tk.StringVar(value="4K (3840x2160)")
        res_combo = ttk.Combobox(row1, textvariable=self.resolution_var,
                                values=list(self.RESOLUTIONS.keys()), width=16, state="readonly")
        res_combo.pack(side=tk.LEFT, padx=(0, 30))
        res_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_video_preview())

        # Transition
        tk.Label(row1, text="Chuyển cảnh:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.transition_var = tk.StringVar(value="Ngẫu nhiên")
        trans_combo = ttk.Combobox(row1, textvariable=self.transition_var,
                                  values=list(self.TRANSITIONS.keys()), width=14, state="readonly")
        trans_combo.pack(side=tk.LEFT)
        trans_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_video_preview())

        # Row 2: Ken Burns
        row2 = tk.Frame(parent, bg=COLORS["bg_card"])
        row2.pack(fill=tk.X, pady=5)

        tk.Label(row2, text="Ken Burns:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.kenburns_var = tk.StringVar(value="Nhẹ")
        kb_combo = ttk.Combobox(row2, textvariable=self.kenburns_var,
                               values=list(self.KEN_BURNS.keys()), width=16, state="readonly")
        kb_combo.pack(side=tk.LEFT)

        tk.Label(row2, text="(hiệu ứng zoom/pan ảnh)", font=("Segoe UI", 9),
                bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(10, 0))

    def _create_nv_content(self, parent):
        """NV Overlay settings section."""
        # Row 1: Enable + Scale
        row1 = tk.Frame(parent, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, pady=5)

        self.nv_enabled_var = tk.BooleanVar(value=True)
        nv_cb = tk.Checkbutton(row1, text="Bật NV Overlay", variable=self.nv_enabled_var,
                              bg=COLORS["bg_card"], fg=COLORS["text"],
                              selectcolor=COLORS["bg_dark"], font=("Segoe UI", 10),
                              activebackground=COLORS["bg_card"], activeforeground=COLORS["text"],
                              command=self._refresh_video_preview)
        nv_cb.pack(side=tk.LEFT)

        tk.Label(row1, text="Kích thước:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=(40, 5))
        self.nv_scale_var = tk.IntVar(value=50)
        scale_slider = tk.Scale(row1, from_=30, to=80, orient=tk.HORIZONTAL,
                               variable=self.nv_scale_var, bg=COLORS["bg_card"],
                               fg=COLORS["text"], highlightthickness=0, length=180,
                               troughcolor=COLORS["bg_dark"], activebackground=COLORS["accent_blue"],
                               command=lambda v: self._refresh_video_preview())
        scale_slider.pack(side=tk.LEFT)
        tk.Label(row1, text="%", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"]).pack(side=tk.LEFT)

        # Row 2: Position
        row2 = tk.Frame(parent, bg=COLORS["bg_card"])
        row2.pack(fill=tk.X, pady=5)

        tk.Label(row2, text="Vị trí ngang:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.nv_h_pos_var = tk.StringVar(value="Trái")
        h_combo = ttk.Combobox(row2, textvariable=self.nv_h_pos_var,
                              values=list(self.NV_H_POSITIONS.keys()), width=10, state="readonly")
        h_combo.pack(side=tk.LEFT, padx=(0, 30))
        h_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_video_preview())

        tk.Label(row2, text="Vị trí dọc:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.nv_v_pos_var = tk.StringVar(value="Giữa")
        v_combo = ttk.Combobox(row2, textvariable=self.nv_v_pos_var,
                              values=list(self.NV_V_POSITIONS.keys()), width=10, state="readonly")
        v_combo.pack(side=tk.LEFT)
        v_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_video_preview())

    def _create_subtitle_content(self, parent):
        """Subtitle settings section."""
        # Row 1: Font + Size
        row1 = tk.Frame(parent, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, pady=5)

        tk.Label(row1, text="Font chữ:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.font_var = tk.StringVar(value="Bebas Neue")
        font_combo = ttk.Combobox(row1, textvariable=self.font_var,
                                 values=self.available_fonts, width=20, state="readonly")
        font_combo.pack(side=tk.LEFT, padx=(0, 30))

        tk.Label(row1, text="Cỡ chữ:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=8, anchor=tk.W).pack(side=tk.LEFT)
        self.size_var = tk.IntVar(value=28)
        size_spin = tk.Spinbox(row1, from_=16, to=60, textvariable=self.size_var,
                              width=5, font=("Segoe UI", 10))
        size_spin.pack(side=tk.LEFT)

        # Row 2: Color + Position
        row2 = tk.Frame(parent, bg=COLORS["bg_card"])
        row2.pack(fill=tk.X, pady=5)

        tk.Label(row2, text="Màu chữ:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value="Trắng")
        color_combo = ttk.Combobox(row2, textvariable=self.color_var,
                                  values=list(self.SUBTITLE_COLORS.keys()), width=10, state="readonly")
        color_combo.pack(side=tk.LEFT)
        color_combo.bind("<<ComboboxSelected>>", self._on_color_changed)

        # Color preview
        self.color_preview = tk.Label(row2, text=" ABC ", font=("Segoe UI", 11, "bold"),
                                     bg="#FFFFFF", fg="#000000", relief=tk.SOLID, borderwidth=1)
        self.color_preview.pack(side=tk.LEFT, padx=(10, 0))

        # Subtitle position
        tk.Label(row2, text="Vị trí:", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=(25, 5))
        self.sub_pos_var = tk.StringVar(value="Dưới")
        sub_pos_combo = ttk.Combobox(row2, textvariable=self.sub_pos_var,
                                    values=list(self.SUBTITLE_POSITIONS.keys()), width=8, state="readonly")
        sub_pos_combo.pack(side=tk.LEFT)
        sub_pos_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_video_preview())

    def _on_color_changed(self, event=None):
        """Handle color change - update preview and video preview."""
        self._update_color_preview()
        self._refresh_video_preview()

    def _refresh_video_preview(self):
        """Refresh the video preview with current settings."""
        if self.current_channel:
            self._show_video_preview(self.current_channel)

    def _update_color_preview(self, event=None):
        """Update color preview when color changes."""
        color_name = self.color_var.get()
        ass_color = self.SUBTITLE_COLORS.get(color_name, "&H00FFFFFF")
        # Convert ASS color to hex (ASS format: &HBBGGRR)
        try:
            bgr = ass_color.replace("&H00", "").replace("&H", "")
            r = int(bgr[4:6], 16)
            g = int(bgr[2:4], 16)
            b = int(bgr[0:2], 16)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            # Text color (contrasting)
            text_color = "#000000" if (r + g + b) > 384 else "#FFFFFF"
            self.color_preview.config(bg=hex_color, fg=text_color)
        except:
            pass

    def on_channel_selected(self, event):
        channel = self.channel_combo.get().strip().upper()
        if not channel:
            return

        self.current_channel = channel
        self.channel_var.set(channel)

        # Load settings if exists
        if channel in self.templates:
            t = self.templates[channel]

            # Video settings
            res_val = t.get("output_resolution", "4k")
            for name, val in self.RESOLUTIONS.items():
                if val == res_val:
                    self.resolution_var.set(name)
                    break

            trans_val = t.get("video_transition", "random")
            for name, val in self.TRANSITIONS.items():
                if val == trans_val:
                    self.transition_var.set(name)
                    break

            kb_val = t.get("ken_burns_intensity", "subtle")
            for name, val in self.KEN_BURNS.items():
                if val == kb_val:
                    self.kenburns_var.set(name)
                    break

            # NV settings
            self.nv_enabled_var.set(t.get("nv_overlay_enabled", True))
            self.nv_scale_var.set(int(t.get("nv_overlay_scale", 0.50) * 100))

            h_pos = t.get("nv_overlay_position", "left")
            for name, val in self.NV_H_POSITIONS.items():
                if val == h_pos:
                    self.nv_h_pos_var.set(name)
                    break

            v_pos = t.get("nv_overlay_v_position", "middle")
            for name, val in self.NV_V_POSITIONS.items():
                if val == v_pos:
                    self.nv_v_pos_var.set(name)
                    break

            # Subtitle settings
            font = t.get("font", "Bebas Neue")
            if font in self.available_fonts:
                self.font_var.set(font)

            self.size_var.set(t.get("size", 28))

            color_val = t.get("color", "&H00FFFFFF")
            for name, val in self.SUBTITLE_COLORS.items():
                if val == color_val:
                    self.color_var.set(name)
                    break
            self._update_color_preview()

            # Subtitle position (alignment)
            align_val = t.get("alignment", 2)
            for name, val in self.SUBTITLE_POSITIONS.items():
                if val == align_val:
                    self.sub_pos_var.set(name)
                    break

            self.app.log(f"Đã tải: {channel}", "success")

        # Show previews
        self._show_real_preview(channel)
        self._show_video_preview(channel)

    def _show_real_preview(self, channel):
        """Show REAL thumbnail of this channel if exists."""
        if not PIL_AVAILABLE:
            self.preview_label.config(text="N/A", image="")
            return

        # Find any thumbnail for this channel
        # Thumbnails are named by image_code (e.g., KA2-0241.jpg), not channel name
        # Extract channel prefix (e.g., KA2 from KA2-T2)
        channel_prefix = channel.split("-")[0] if "-" in channel else channel

        thumb_path = None
        # Search for thumbnails with channel prefix (KA2-*.jpg matches KA2-0241.jpg)
        for pattern in [f"{channel_prefix}-*.jpg", f"{channel_prefix}-*.png",
                       f"{channel_prefix.lower()}-*.jpg", f"{channel_prefix.lower()}-*.png"]:
            matches = list(self.THUMB_OUTPUT_DIR.glob(pattern))
            if matches:
                # Get most recent thumbnail
                thumb_path = max(matches, key=lambda f: f.stat().st_mtime)
                break

        if not thumb_path:
            self.preview_label.config(text=f"Chưa có\nthumbnail", image="", fg=COLORS["text_dim"])
            return

        try:
            img = Image.open(thumb_path)
            # Resize to fit preview (max 280x130)
            ratio = min(280 / img.width, 130 / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

            self.preview_image = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.preview_image, text="")
        except Exception as e:
            self.preview_label.config(text=f"Lỗi", image="")

    def _show_video_preview(self, channel):
        """Generate a realistic preview showing NV position and subtitle style."""
        if not PIL_AVAILABLE:
            self.video_preview_label.config(text="N/A", image="")
            return

        try:
            from PIL import ImageDraw, ImageFont

            # Preview size (16:9 ratio)
            width, height = 320, 180

            # Try to find a background image from pic folder
            pic_dir = TOOL_DIR / "thumb" / "pic"
            bg_img = None
            if pic_dir.exists():
                # Find any jpg/png in pic folder
                for pattern in ["*.jpg", "*.png", "*.jpeg"]:
                    pics = list(pic_dir.glob(pattern))
                    if pics:
                        try:
                            bg_img = Image.open(pics[0])
                            # Resize and crop to 16:9
                            bg_ratio = bg_img.width / bg_img.height
                            target_ratio = width / height
                            if bg_ratio > target_ratio:
                                new_h = bg_img.height
                                new_w = int(new_h * target_ratio)
                                left = (bg_img.width - new_w) // 2
                                bg_img = bg_img.crop((left, 0, left + new_w, new_h))
                            else:
                                new_w = bg_img.width
                                new_h = int(new_w / target_ratio)
                                top = (bg_img.height - new_h) // 2
                                bg_img = bg_img.crop((0, top, new_w, top + new_h))
                            bg_img = bg_img.resize((width, height), Image.LANCZOS)
                            break
                        except:
                            continue

            # Create base image
            if bg_img:
                img = bg_img.convert("RGB")
            else:
                img = Image.new("RGB", (width, height), "#2a2a3e")

            draw = ImageDraw.Draw(img)

            # Get current settings
            nv_enabled = self.nv_enabled_var.get()
            nv_h_pos = self.NV_H_POSITIONS.get(self.nv_h_pos_var.get(), "left")
            nv_v_pos = self.NV_V_POSITIONS.get(self.nv_v_pos_var.get(), "middle")
            nv_scale = self.nv_scale_var.get() / 100.0

            # Draw NV overlay
            if nv_enabled:
                # Find actual NV image for this channel
                channel_prefix = channel.split("-")[0] if "-" in channel else channel
                nv_path = None

                # Try exact channel match first
                exact_nv = self.NV_OUTPUT_DIR / f"{channel}.png"
                if exact_nv.exists():
                    nv_path = exact_nv
                else:
                    # Try any NV with channel prefix
                    for f in self.NV_OUTPUT_DIR.glob(f"{channel_prefix}-*.png"):
                        nv_path = f
                        break

                if nv_path and nv_path.exists():
                    try:
                        nv_img = Image.open(nv_path).convert("RGBA")

                        # Scale NV to configured size
                        nv_height = int(height * nv_scale)
                        nv_ratio = nv_img.width / nv_img.height
                        nv_width = int(nv_height * nv_ratio)
                        nv_img = nv_img.resize((nv_width, nv_height), Image.LANCZOS)

                        # Calculate position
                        margin = 8
                        if nv_h_pos == "left":
                            nv_x = margin
                        else:
                            nv_x = width - nv_width - margin

                        if nv_v_pos == "top":
                            nv_y = margin
                        elif nv_v_pos == "middle":
                            nv_y = (height - nv_height) // 2
                        else:  # bottom
                            nv_y = height - nv_height - margin - 20

                        # Paste NV with alpha
                        img.paste(nv_img, (nv_x, nv_y), nv_img)
                    except Exception as e:
                        pass

            # Draw subtitle preview
            sub_color = self.SUBTITLE_COLORS.get(self.color_var.get(), "&H00FFFFFF")
            try:
                bgr = sub_color.replace("&H00", "").replace("&H", "")
                r = int(bgr[4:6], 16)
                g = int(bgr[2:4], 16)
                b = int(bgr[0:2], 16)
                text_color = (r, g, b)
            except:
                text_color = (255, 255, 255)

            # Try to load actual font
            font = None
            font_name = self.font_var.get()
            font_size = max(10, self.size_var.get() // 3)  # Scale down for preview
            try:
                for f in self.FONTS_DIR.glob("*.ttf"):
                    if font_name.lower().replace(" ", "") in f.stem.lower().replace("-", "").replace("_", ""):
                        font = ImageFont.truetype(str(f), font_size)
                        break
            except:
                pass

            # Get subtitle position (with proper margins)
            sub_pos = self.sub_pos_var.get()
            if sub_pos == "Trên":
                sub_y = 22
            elif sub_pos == "Giữa":
                sub_y = height // 2
            else:  # Dưới
                sub_y = height - 30  # More margin from bottom

            # Draw subtitle text with outline
            sub_text = "Phụ đề mẫu - Sample subtitle"

            # Draw black outline
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((width//2 + dx, sub_y + dy), sub_text,
                                 fill=(0, 0, 0), font=font, anchor="mm")
            # Draw main text
            draw.text((width//2, sub_y), sub_text, fill=text_color, font=font, anchor="mm")

            # Draw settings badge (semi-transparent)
            info_text = f"{self.resolution_var.get().split()[0]} | {self.transition_var.get()}"
            badge_w = len(info_text) * 6 + 10
            draw.rectangle([2, 2, badge_w, 16], fill=(0, 0, 0, 128))
            draw.text((5, 3), info_text, fill=(200, 200, 200))

            self.video_preview_image = ImageTk.PhotoImage(img)
            self.video_preview_label.config(image=self.video_preview_image, text="")

        except Exception as e:
            self.video_preview_label.config(text=f"Lỗi: {e}", image="")

    def save_template(self):
        channel = self.channel_var.get().strip().upper()
        if not channel:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập tên channel!")
            return

        # Build template from all settings
        template = self.DEFAULT_TEMPLATE.copy()

        # Video
        template["output_resolution"] = self.RESOLUTIONS.get(self.resolution_var.get(), "4k")
        template["video_transition"] = self.TRANSITIONS.get(self.transition_var.get(), "random")
        template["ken_burns_intensity"] = self.KEN_BURNS.get(self.kenburns_var.get(), "subtle")

        # NV Overlay
        template["nv_overlay_enabled"] = self.nv_enabled_var.get()
        template["nv_overlay_scale"] = self.nv_scale_var.get() / 100.0
        template["nv_overlay_position"] = self.NV_H_POSITIONS.get(self.nv_h_pos_var.get(), "left")
        template["nv_overlay_v_position"] = self.NV_V_POSITIONS.get(self.nv_v_pos_var.get(), "middle")

        # Subtitle
        template["font"] = self.font_var.get()
        template["size"] = self.size_var.get()
        template["color"] = self.SUBTITLE_COLORS.get(self.color_var.get(), "&H00FFFFFF")
        template["alignment"] = self.SUBTITLE_POSITIONS.get(self.sub_pos_var.get(), 2)

        self.templates[channel] = template
        self.save_templates()
        self.app.log(f"Đã lưu: {channel}", "success")

        # Update dropdown
        self.channel_combo['values'] = list(self.templates.keys())
        messagebox.showinfo("✓", f"Đã lưu cài đặt cho {channel}!")

    def delete_template(self):
        channel = self.channel_var.get().strip().upper()
        if not channel:
            return

        if channel in self.templates:
            if messagebox.askyesno("Xác nhận", f"Xóa cài đặt của {channel}?"):
                del self.templates[channel]
                self.save_templates()
                self.channel_combo['values'] = list(self.templates.keys())
                self.channel_var.set("")
                self.app.log(f"Đã xóa: {channel}", "warning")


# ============================================================================
# MAIN
# ============================================================================

def main():
    root = tk.Tk()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = VE3ToolGUI(root)

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
