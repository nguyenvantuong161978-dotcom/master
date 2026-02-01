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


class StatCard(tk.Frame):
    """Modern stat card with icon."""

    def __init__(self, parent, icon, title, value, color, **kwargs):
        super().__init__(parent, bg=COLORS["bg_card"], **kwargs)
        self.config(highlightbackground=COLORS["border"], highlightthickness=1)

        inner = tk.Frame(self, bg=COLORS["bg_card"])
        inner.pack(padx=20, pady=15, fill=tk.BOTH, expand=True)

        top_row = tk.Frame(inner, bg=COLORS["bg_card"])
        top_row.pack(fill=tk.X)

        tk.Label(top_row, text=icon, font=("Segoe UI", 24),
                bg=COLORS["bg_card"], fg=color).pack(side=tk.LEFT)

        self.value_label = tk.Label(top_row, text=value, font=("Segoe UI", 28, "bold"),
                                    bg=COLORS["bg_card"], fg=color)
        self.value_label.pack(side=tk.RIGHT)

        tk.Label(inner, text=title, font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(anchor=tk.W, pady=(10, 0))

    def set_value(self, value):
        self.value_label.config(text=str(value))


class ProcessCard(tk.Frame):
    """Card for process control with progress bar."""

    def __init__(self, parent, title, subtitle, icon, color, on_start, on_stop, **kwargs):
        super().__init__(parent, bg=COLORS["bg_card"], **kwargs)
        self.config(highlightbackground=COLORS["border"], highlightthickness=1)
        self.color = color
        self.on_start = on_start
        self.on_stop = on_stop

        inner = tk.Frame(self, bg=COLORS["bg_card"])
        inner.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        header = tk.Frame(inner, bg=COLORS["bg_card"])
        header.pack(fill=tk.X)

        tk.Label(header, text=icon, font=("Segoe UI", 20),
                bg=COLORS["bg_card"], fg=color).pack(side=tk.LEFT)

        title_frame = tk.Frame(header, bg=COLORS["bg_card"])
        title_frame.pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(title_frame, text=title, font=("Segoe UI", 12, "bold"),
                bg=COLORS["bg_card"], fg=COLORS["text"]).pack(anchor=tk.W)
        tk.Label(title_frame, text=subtitle, font=("Segoe UI", 9),
                bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(anchor=tk.W)

        status_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        status_frame.pack(fill=tk.X, pady=(15, 0))

        self.status_dot = tk.Label(status_frame, text="●", font=("Segoe UI", 12),
                                   bg=COLORS["bg_card"], fg=COLORS["error"])
        self.status_dot.pack(side=tk.LEFT)

        self.status_text = tk.Label(status_frame, text="Stopped", font=("Segoe UI", 10),
                                    bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.status_text.pack(side=tk.LEFT, padx=(5, 0))

        # Progress section
        progress_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        self.progress_label = tk.Label(progress_frame, text="", font=("Segoe UI", 9),
                                       bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.progress_label.pack(anchor=tk.W)

        # Progress bar
        self.progress_canvas = tk.Canvas(progress_frame, height=8, bg=COLORS["bg_dark"],
                                        highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X, pady=(5, 0))

        self.percent_label = tk.Label(progress_frame, text="", font=("Segoe UI", 9, "bold"),
                                      bg=COLORS["bg_card"], fg=color)
        self.percent_label.pack(anchor=tk.E)

        btn_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        btn_frame.pack(fill=tk.X, pady=(15, 0))

        self.start_btn = ModernButton(btn_frame, "▶ Start", self._on_start,
                                      color, width=100, height=36)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ModernButton(btn_frame, "■ Stop", self._on_stop,
                                     COLORS["accent_red"], width=100, height=36)
        self.stop_btn.pack(side=tk.LEFT)
        self.stop_btn.set_disabled(True)

    def _on_start(self):
        if self.on_start:
            self.on_start()

    def _on_stop(self):
        if self.on_stop:
            self.on_stop()

    def set_running(self, running):
        if running:
            self.status_dot.config(fg=COLORS["success"])
            self.status_text.config(text="Running", fg=COLORS["success"])
            self.start_btn.set_disabled(True)
            self.stop_btn.set_disabled(False)
        else:
            self.status_dot.config(fg=COLORS["error"])
            self.status_text.config(text="Stopped", fg=COLORS["text_dim"])
            self.start_btn.set_disabled(False)
            self.stop_btn.set_disabled(True)
            # Clear progress
            self.set_progress(0, "", "")

    def set_progress(self, percent, code, step):
        """Update progress bar and labels."""
        # Update progress bar
        self.progress_canvas.delete("all")
        width = self.progress_canvas.winfo_width()
        if width > 1:
            # Background
            self.progress_canvas.create_rectangle(0, 0, width, 8,
                                                  fill=COLORS["bg_dark"], outline="")
            # Fill
            fill_width = int(width * percent / 100)
            if fill_width > 0:
                self.progress_canvas.create_rectangle(0, 0, fill_width, 8,
                                                      fill=self.color, outline="")

        # Update labels
        if code and step:
            self.progress_label.config(text=f"{code}: {step}")
            self.percent_label.config(text=f"{percent}%")
        else:
            self.progress_label.config(text="")
            self.percent_label.config(text="")


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
        self.parallel_var = tk.StringVar(value="4")  # Default parallel count

        # Smart queue tracker
        self.queue_tracker = QueueTracker()

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
        main = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        self.create_header(main)
        self.create_progress_bar(main)
        self.create_code_list(main)
        self.create_log_area(main)

    def create_header(self, parent):
        """Header với Auto button và Settings."""
        header = tk.Frame(parent, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 15))

        # Title
        tk.Label(header, text="⚡ VE3 Tool", font=("Segoe UI", 18, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT)

        # Settings button
        settings_btn = tk.Button(header, text="⚙ Cài đặt", font=("Segoe UI", 10),
                                command=self.open_subtitle_settings,
                                bg=COLORS["bg_card"], fg=COLORS["text"],
                                relief=tk.FLAT, cursor="hand2", padx=12, pady=5)
        settings_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Refresh button
        refresh_btn = tk.Button(header, text="🔄", font=("Segoe UI", 12),
                               command=self.refresh_all,
                               bg=COLORS["bg_dark"], fg=COLORS["text_dim"],
                               relief=tk.FLAT, cursor="hand2", padx=8)
        refresh_btn.pack(side=tk.RIGHT)

        # Auto mode button
        self.auto_btn = ModernButton(header, "▶ Chạy Auto", self.toggle_auto_mode,
                                    COLORS["accent_green"], width=110, height=36)
        self.auto_btn.pack(side=tk.RIGHT, padx=(0, 10))

    def create_progress_bar(self, parent):
        """Progress bar lớn và rõ ràng khi đang xử lý."""
        self.progress_frame = tk.Frame(parent, bg=COLORS["accent_blue"],
                                       highlightbackground=COLORS["accent_blue"], highlightthickness=2)
        self.progress_frame.pack(fill=tk.X, pady=(0, 15))
        self.progress_frame.pack_forget()  # Ẩn khi không chạy

        inner = tk.Frame(self.progress_frame, bg=COLORS["bg_card"])
        inner.pack(fill=tk.BOTH, padx=2, pady=2)

        # Row 1: Code và phần trăm (lớn)
        row1 = tk.Frame(inner, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, padx=15, pady=(12, 5))

        self.progress_code_label = tk.Label(row1, text="", font=("Segoe UI", 16, "bold"),
                                           bg=COLORS["bg_card"], fg=COLORS["accent_blue"])
        self.progress_code_label.pack(side=tk.LEFT)

        self.progress_percent_label = tk.Label(row1, text="", font=("Segoe UI", 18, "bold"),
                                              bg=COLORS["bg_card"], fg=COLORS["success"])
        self.progress_percent_label.pack(side=tk.RIGHT)

        # Row 2: Step chi tiết
        self.progress_step_label = tk.Label(inner, text="", font=("Segoe UI", 11),
                                           bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.progress_step_label.pack(fill=tk.X, padx=15, pady=(0, 8))

        # Progress bar lớn hơn
        self.progress_bar = tk.Canvas(inner, height=12, bg=COLORS["bg_dark"],
                                     highlightthickness=0)
        self.progress_bar.pack(fill=tk.X, padx=15, pady=(0, 12))

    def create_code_list(self, parent):
        """Danh sách mã với trạng thái rõ ràng."""
        # Container
        list_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Tạo 3 cột: Chờ VM | Chờ Edit | Hoàn thành
        list_frame.columnconfigure(0, weight=1)
        list_frame.columnconfigure(1, weight=1)
        list_frame.columnconfigure(2, weight=1)

        # Column 1: Chờ VM (orange)
        self.vm_column = self._create_status_column(list_frame, 0, "📤 Chờ VM", COLORS["accent_orange"])

        # Column 2: Chờ Edit (blue)
        self.edit_column = self._create_status_column(list_frame, 1, "🎬 Chờ Edit", COLORS["accent_blue"])

        # Column 3: Hoàn thành (green)
        self.done_column = self._create_status_column(list_frame, 2, "✅ Hoàn thành", COLORS["success"])

        # Initial load
        self.refresh_code_list()

    def _create_status_column(self, parent, col, title, color):
        """Tạo một cột trạng thái."""
        frame = tk.Frame(parent, bg=COLORS["bg_card"], highlightbackground=COLORS["border"],
                        highlightthickness=1)
        frame.grid(row=0, column=col, sticky="nsew", padx=5)

        # Header
        header = tk.Frame(frame, bg=COLORS["bg_card"])
        header.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(header, text=title, font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_card"], fg=color).pack(side=tk.LEFT)

        count_label = tk.Label(header, text="0", font=("Segoe UI", 14, "bold"),
                              bg=COLORS["bg_card"], fg=color)
        count_label.pack(side=tk.RIGHT)

        # List container with scrollbar
        list_container = tk.Frame(frame, bg=COLORS["bg_card"])
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        canvas = tk.Canvas(list_container, bg=COLORS["bg_card"], highlightthickness=0, height=280)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)

        list_frame = tk.Frame(canvas, bg=COLORS["bg_card"])
        list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=list_frame, anchor="nw", width=250)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        list_frame.bind("<MouseWheel>", on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        return {"frame": list_frame, "count": count_label, "canvas": canvas, "color": color}

    def refresh_code_list(self):
        """Refresh danh sách mã theo trạng thái."""
        # Get all codes
        all_codes = self.queue_tracker.get_all_codes()

        # Read current progress to mark in-progress items
        current_code = ""
        current_step = ""
        current_percent = 0
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                current_code = progress.get("code", "")
                current_step = progress.get("step", "")
                current_percent = progress.get("percent", 0)
                clip_current = progress.get("clip_current", 0)
                clip_total = progress.get("clip_total", 0)
        except:
            pass

        # Update in-progress item
        if current_code and current_code in all_codes:
            if current_step and current_percent > 0:
                all_codes[current_code]["status"] = "editing"
                # Show detailed progress
                if clip_total > 0:
                    all_codes[current_code]["status_text"] = f"{current_step} {clip_current}/{clip_total} ({current_percent}%)"
                else:
                    all_codes[current_code]["status_text"] = f"{current_step} ({current_percent}%)"
                all_codes[current_code]["icon"] = "⏳"
                all_codes[current_code]["color"] = COLORS["warning"]

        # Clear all columns
        for col in [self.vm_column, self.edit_column, self.done_column]:
            for widget in col["frame"].winfo_children():
                widget.destroy()

        # Categorize codes
        vm_codes = []      # voice, srt_done, waiting_vm
        edit_codes = []    # visual_ready, editing
        done_codes = []    # done

        for code, data in all_codes.items():
            status = data.get("status", "")
            if status in ["voice", "srt_done", "waiting_vm"]:
                vm_codes.append((code, data))
            elif status in ["visual_ready", "editing"]:
                edit_codes.append((code, data))
            elif status == "done":
                done_codes.append((code, data))

        # Sort edit_codes: editing first, then visual_ready
        edit_codes.sort(key=lambda x: (0 if x[1].get("status") == "editing" else 1, x[0]))

        # Sort done by mtime (newest first)
        done_codes.sort(key=lambda x: -x[1].get("mtime", 0))

        # Populate columns
        self._populate_column(self.vm_column, vm_codes)
        self._populate_column(self.edit_column, edit_codes)
        self._populate_column(self.done_column, done_codes[:20])  # Max 20 done items

        # Update counts
        self.vm_column["count"].config(text=str(len(vm_codes)))
        self.edit_column["count"].config(text=str(len(edit_codes)))
        self.done_column["count"].config(text=str(len(done_codes)))

    def _populate_column(self, column, items):
        """Populate a column with code items."""
        frame = column["frame"]
        color = column["color"]

        for code, data in items:
            is_editing = data.get("status") == "editing"
            bg_color = COLORS["accent_blue"] if is_editing else COLORS["bg_card"]
            text_color = "#ffffff" if is_editing else COLORS["text"]

            item = tk.Frame(frame, bg=bg_color)
            item.pack(fill=tk.X, pady=2)

            # Status icon
            icon = data.get("icon", "●")
            tk.Label(item, text=icon, font=("Segoe UI", 10),
                    bg=bg_color, fg="#ffffff" if is_editing else color).pack(side=tk.LEFT, padx=(5, 5))

            # Code name (bold if editing)
            font_style = ("Segoe UI", 11, "bold") if is_editing else ("Segoe UI", 10)
            code_label = tk.Label(item, text=code, font=font_style,
                                 bg=bg_color, fg=text_color)
            code_label.pack(side=tk.LEFT)

            # Status text (smaller)
            status_text = data.get("status_text", "")
            if status_text and "Hoàn thành" not in status_text:
                tk.Label(item, text=f"• {status_text}", font=("Segoe UI", 8),
                        bg=bg_color, fg="#cccccc" if is_editing else COLORS["text_dim"]).pack(side=tk.LEFT, padx=(5, 0))

            # For done items: show file icons and click to open folder
            if data.get("status") == "done":
                video_path = data.get("path")
                done_folder = video_path.parent if video_path else None

                # Check what files exist
                thumb_dir = TOOL_DIR / "thumb" / "thumbnails"
                has_thumb = any(thumb_dir.glob(f"{code}.*")) if thumb_dir.exists() else False
                has_video = video_path and video_path.exists()
                has_srt = done_folder and any(done_folder.glob("*.srt")) if done_folder else False

                # Show file icons
                icons_text = ""
                if has_thumb:
                    icons_text += "🖼"
                if has_video:
                    icons_text += "🎬"
                if has_srt:
                    icons_text += "📝"

                if icons_text:
                    tk.Label(item, text=icons_text, font=("Segoe UI", 8),
                            bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.RIGHT, padx=(0, 5))

                # Click to open folder, double-click to play video
                item.config(cursor="hand2")
                code_label.config(cursor="hand2")

                if done_folder:
                    item.bind("<Button-1>", lambda e, p=done_folder: os.startfile(str(p)))
                    code_label.bind("<Button-1>", lambda e, p=done_folder: os.startfile(str(p)))
                if video_path:
                    item.bind("<Double-Button-1>", lambda e, p=video_path: self.play_video(p))
                    code_label.bind("<Double-Button-1>", lambda e, p=video_path: self.play_video(p))

                item.bind("<Enter>", lambda e, f=item: f.configure(bg=COLORS["bg_card_hover"]))
                item.bind("<Leave>", lambda e, f=item: f.configure(bg=COLORS["bg_card"]))

    def refresh_all(self):
        """Refresh cả danh sách và stats."""
        self.refresh_code_list()
        self.log("Đã làm mới", "info")

    def create_status_bar(self, parent):
        """Compact status bar showing current processing state."""
        status_frame = tk.Frame(parent, bg=COLORS["bg_card"],
                               highlightbackground=COLORS["border"], highlightthickness=1)
        status_frame.pack(fill=tk.X, pady=(0, 15))

        inner = tk.Frame(status_frame, bg=COLORS["bg_card"])
        inner.pack(fill=tk.X, padx=15, pady=10)

        # SRT status
        srt_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        srt_frame.pack(side=tk.LEFT, padx=(0, 20))

        self.srt_status_dot = tk.Label(srt_frame, text="●", font=("Segoe UI", 10),
                                       bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.srt_status_dot.pack(side=tk.LEFT)
        tk.Label(srt_frame, text="SRT", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=(5, 0))

        # Edit status
        edit_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        edit_frame.pack(side=tk.LEFT, padx=(0, 20))

        self.edit_status_dot = tk.Label(edit_frame, text="●", font=("Segoe UI", 10),
                                        bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.edit_status_dot.pack(side=tk.LEFT)
        tk.Label(edit_frame, text="Edit", font=("Segoe UI", 10),
                bg=COLORS["bg_card"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=(5, 0))

        # Parallel selector
        parallel_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        parallel_frame.pack(side=tk.LEFT, padx=(0, 20))
        tk.Label(parallel_frame, text="Parallel:", font=("Segoe UI", 9),
                bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
        self.parallel_var = tk.StringVar(value="4")
        parallel_combo = ttk.Combobox(parallel_frame, textvariable=self.parallel_var,
                                      values=["1", "2", "3", "4", "6", "8"], width=3, state="readonly")
        parallel_combo.pack(side=tk.LEFT, padx=(5, 0))

        # Current progress (inline in control bar)
        self.inline_progress_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        self.inline_progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        self.current_progress_label = tk.Label(self.inline_progress_frame, text="",
                                               font=("Segoe UI", 9),
                                               bg=COLORS["bg_card"], fg=COLORS["warning"])
        self.current_progress_label.pack(side=tk.LEFT)

    def create_quick_actions(self, parent):
        actions = tk.Frame(parent, bg=COLORS["bg_dark"])
        actions.pack(fill=tk.X, pady=(0, 15))

        tk.Label(actions, text="Quick Actions", font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(anchor=tk.W, pady=(0, 10))

        btn_row = tk.Frame(actions, bg=COLORS["bg_dark"])
        btn_row.pack(fill=tk.X)

        buttons = [
            ("📂 VISUAL", lambda: os.startfile(VISUAL_DIR) if VISUAL_DIR.exists() else None, COLORS["bg_card_hover"]),
            ("📂 DONE", lambda: os.startfile(DONE_DIR) if DONE_DIR.exists() else None, COLORS["bg_card_hover"]),
            ("📂 VOICE", lambda: os.startfile(VOICE_DIR) if VOICE_DIR.exists() else None, COLORS["bg_card_hover"]),
            ("⚙ Template", self.open_subtitle_settings, COLORS["accent_orange"]),
            ("⬆ Upload", self.upload_github, COLORS["accent_purple"]),
        ]

        for text, cmd, color in buttons:
            btn = ModernButton(btn_row, text, cmd, color, width=95, height=32)
            btn.pack(side=tk.LEFT, padx=(0, 8))

    def create_processing_queue(self, parent):
        """Create processing queue showing pending/in-progress/completed items."""
        queue_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        queue_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Header
        header = tk.Frame(queue_frame, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 5))

        tk.Label(header, text="📋 Trạng thái mã", font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT)

        # Legend
        legend = tk.Label(header, text="🎤SRT  📤VM copy  🖼️VM ảnh  🎬Edit  ✅Done",
                         font=("Segoe UI", 8), bg=COLORS["bg_dark"], fg=COLORS["text_dim"])
        legend.pack(side=tk.LEFT, padx=(15, 0))

        refresh_btn = tk.Label(header, text="Refresh", font=("Segoe UI", 9),
                              bg=COLORS["bg_dark"], fg=COLORS["accent_blue"], cursor="hand2")
        refresh_btn.pack(side=tk.RIGHT)
        refresh_btn.bind("<Button-1>", lambda e: self.refresh_queue())

        # List container
        list_container = tk.Frame(queue_frame, bg=COLORS["border"])
        list_container.pack(fill=tk.BOTH, expand=True)

        # Canvas with scrollbar for the list
        canvas = tk.Canvas(list_container, bg=COLORS["bg_card"], highlightthickness=0, height=180)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)

        self.queue_list_frame = tk.Frame(canvas, bg=COLORS["bg_card"])

        self.queue_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.queue_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.queue_canvas = canvas
        self.refresh_queue()

    def refresh_queue(self):
        """Refresh the processing queue list with smart state tracking."""
        # Clear existing items
        for widget in self.queue_list_frame.winfo_children():
            widget.destroy()

        # Get all codes with smart state detection
        all_codes = self.queue_tracker.get_all_codes()

        # Read current progress to mark in-progress items
        current_code = ""
        current_step = ""
        current_percent = 0
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                current_code = progress.get("code", "")
                current_step = progress.get("step", "")
                current_percent = progress.get("percent", 0)
        except:
            pass

        # Update in-progress item
        if current_code and current_code in all_codes:
            if current_step and current_percent > 0:
                all_codes[current_code]["status"] = "editing"
                all_codes[current_code]["status_text"] = f"{current_step} ({current_percent}%)"
                all_codes[current_code]["icon"] = "⏳"
                all_codes[current_code]["color"] = COLORS["warning"]

        # Sort by status priority and then by code
        status_priority = {
            "editing": 0,      # Đang xử lý - cao nhất
            "voice": 1,        # Chờ SRT
            "srt_done": 2,     # Chờ VM copy
            "waiting_vm": 3,   # Chờ VM tạo ảnh
            "visual_ready": 4, # Chờ Edit
            "done": 5          # Hoàn thành - thấp nhất
        }

        queue_items = []
        for code, data in all_codes.items():
            item = {"code": code, **data}
            queue_items.append(item)

        # Sort: by priority, then by mtime (for done), then by code
        queue_items.sort(key=lambda x: (
            status_priority.get(x["status"], 99),
            -x.get("mtime", 0),  # Newer done items first
            x["code"]
        ))

        if not queue_items:
            tk.Label(self.queue_list_frame, text="Không có mã nào trong hàng đợi",
                    bg=COLORS["bg_card"], fg=COLORS["text_dim"],
                    font=("Segoe UI", 9)).pack(pady=10)
            return

        # Group by status for display
        # Show max 30 items total
        displayed = 0
        max_display = 30

        for item in queue_items:
            if displayed >= max_display:
                break
            self._create_queue_item(item)
            displayed += 1

        # Show count summary at bottom
        status_counts = {}
        for item in queue_items:
            st = item["status"]
            status_counts[st] = status_counts.get(st, 0) + 1

        summary_parts = []
        if status_counts.get("voice", 0):
            summary_parts.append(f"🎤{status_counts['voice']}")
        if status_counts.get("srt_done", 0):
            summary_parts.append(f"📤{status_counts['srt_done']}")
        if status_counts.get("waiting_vm", 0):
            summary_parts.append(f"🖼️{status_counts['waiting_vm']}")
        if status_counts.get("visual_ready", 0):
            summary_parts.append(f"🎬{status_counts['visual_ready']}")
        if status_counts.get("done", 0):
            summary_parts.append(f"✅{status_counts['done']}")

        if summary_parts and len(queue_items) > 5:
            summary_frame = tk.Frame(self.queue_list_frame, bg=COLORS["bg_card"])
            summary_frame.pack(fill=tk.X, padx=10, pady=(5, 2))
            tk.Label(summary_frame, text=" | ".join(summary_parts),
                    font=("Segoe UI", 8), bg=COLORS["bg_card"],
                    fg=COLORS["text_dim"]).pack(side=tk.RIGHT)

    def _create_queue_item(self, item):
        """Create a queue item row."""
        item_frame = tk.Frame(self.queue_list_frame, bg=COLORS["bg_card"])
        item_frame.pack(fill=tk.X, padx=10, pady=2)

        # Get icon from item or use default
        icon = item.get("icon", "●")

        # Code
        code_label = tk.Label(item_frame, text=f"{icon} {item['code']}",
                             font=("Segoe UI", 10), bg=COLORS["bg_card"],
                             fg=item["color"])
        code_label.pack(side=tk.LEFT)

        # Status text
        status_label = tk.Label(item_frame, text=item["status_text"],
                               font=("Segoe UI", 9), bg=COLORS["bg_card"],
                               fg=COLORS["text_dim"])
        status_label.pack(side=tk.RIGHT)

        # Click to play if done
        if item["status"] == "done" and "path" in item and item["path"]:
            item_frame.config(cursor="hand2")
            code_label.config(cursor="hand2")
            video_path = item["path"]
            for widget in [item_frame, code_label]:
                widget.bind("<Button-1>", lambda e, p=video_path: self.play_video(p))
                widget.bind("<Enter>", lambda e, f=item_frame: f.configure(bg=COLORS["bg_card_hover"]))
                widget.bind("<Leave>", lambda e, f=item_frame: f.configure(bg=COLORS["bg_card"]))

    def play_video(self, video_path):
        """Play video file."""
        try:
            os.startfile(str(video_path))
            self.log(f"Playing: {video_path.name}", "info")
        except Exception as e:
            self.log(f"Cannot play video: {e}", "error")

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
        """Read progress file and update progress bar."""
        try:
            if PROGRESS_FILE.exists() and self.edit_running:
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    progress = json.load(f)

                code = progress.get("code", "")
                step = progress.get("step", "")
                percent = progress.get("percent", 0)
                clip_current = progress.get("clip_current", 0)
                clip_total = progress.get("clip_total", 0)

                # Add clip info to step if creating clips
                if clip_total > 0 and "clip" in step.lower():
                    step = f"{step} ({clip_current}/{clip_total})"

                # Update progress bar
                self.show_progress(code, step, percent)

                # Track current code to detect changes
                if not hasattr(self, '_last_progress_code'):
                    self._last_progress_code = ""
                if not hasattr(self, '_last_refresh_time'):
                    self._last_refresh_time = 0

                # Refresh list when code changes or every 5 seconds
                now = time.time()
                if code != self._last_progress_code or (now - self._last_refresh_time) > 5:
                    self._last_progress_code = code
                    self._last_refresh_time = now
                    self.refresh_code_list()

                # Also refresh when video is done
                if step == "Done" and percent == 100:
                    self.root.after(1000, self.refresh_code_list)
            else:
                # Hide progress when not running
                if not self.edit_running:
                    self.hide_progress()
        except:
            pass

        # Refresh every 1000ms (giảm CPU)
        self.root.after(1000, self.update_progress)

    def show_progress(self, code, step, percent):
        """Show and update progress bar."""
        if code and step:
            self.progress_frame.pack(fill=tk.X, pady=(0, 15), before=self.vm_column["frame"].master.master)
            self.progress_code_label.config(text=f"⚡ ĐANG XỬ LÝ: {code}")
            self.progress_step_label.config(text=f"📊 {step}")
            self.progress_percent_label.config(text=f"{percent}%")

            # Update progress bar (height=12)
            self.progress_bar.delete("all")
            width = self.progress_bar.winfo_width()
            if width > 1:
                fill_width = int(width * percent / 100)
                if fill_width > 0:
                    self.progress_bar.create_rectangle(0, 0, fill_width, 12,
                                                      fill=COLORS["accent_green"], outline="")

    def hide_progress(self):
        """Hide progress bar."""
        try:
            self.progress_frame.pack_forget()
        except:
            pass

    def refresh_stats(self):
        """Refresh stats và code list."""
        self.refresh_code_list()
        self.root.after(10000, self.refresh_stats)

    def toggle_auto_mode(self):
        """Toggle auto mode - runs SRT, Thumb/NV, and Edit continuously."""
        if self.auto_mode:
            # Stop auto mode
            self.auto_mode = False
            self.auto_btn.text = "▶ Chạy Auto"
            self.auto_btn.color = COLORS["accent_green"]
            self.auto_btn._draw(COLORS["accent_green"])
            self.log("Auto mode stopped", "warning")
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
                        self.root.after(0, lambda l=line, lv=level: self.log(f"[SRT] {l}", lv))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"[SRT] Error: {e}", "error"))
            finally:
                self.root.after(0, self._on_srt_stopped)

        threading.Thread(target=run, daemon=True).start()

    def stop_srt(self):
        if self.srt_process:
            self.log("Stopping SRT Generator...", "warning")
            self.srt_process.terminate()

    def _on_srt_stopped(self):
        self.srt_running = False
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
                        self.root.after(0, lambda l=line, lv=level: self.log(f"[THUMB] {l}", lv))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"[THUMB] Error: {e}", "error"))
            finally:
                self.root.after(0, self._on_thumb_stopped)

        threading.Thread(target=run, daemon=True).start()

    def stop_thumb(self):
        if self.thumb_process:
            self.log("Stopping Thumb/NV Generator...", "warning")
            self.thumb_process.terminate()

    def _on_thumb_stopped(self):
        self.thumb_running = False
        self.log("Thumb/NV Generator stopped", "warning")
        self.refresh_code_list()
        # Restart if auto mode is on
        if self.auto_mode:
            self.root.after(5000, self.start_thumb)  # Wait 5s then restart

    def start_edit(self):
        if self.edit_running:
            return

        parallel = self.parallel_var.get()
        self.log(f"Starting Video Editor (parallel={parallel})...", "info")
        self.edit_running = True

        def run():
            try:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                # Set UTF-8 encoding for subprocess
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                self.edit_process = subprocess.Popen(
                    [sys.executable, str(TOOL_DIR / "run_edit.py"), "--parallel", parallel],
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
                        self.root.after(0, lambda l=line, lv=level: self.log(f"[EDIT] {l}", lv))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"[EDIT] Error: {e}", "error"))
            finally:
                self.root.after(0, self._on_edit_stopped)

        threading.Thread(target=run, daemon=True).start()

    def stop_edit(self):
        if self.edit_process:
            self.log("Stopping Video Editor...", "warning")
            self.edit_process.terminate()

    def _on_edit_stopped(self):
        self.edit_running = False
        self.log("Video Editor stopped", "warning")
        self.refresh_code_list()
        # Restart if auto mode is on
        if self.auto_mode:
            self.root.after(3000, self.start_edit)  # Wait 3s then restart

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
