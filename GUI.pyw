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

# ============================================================================
# CONFIG
# ============================================================================

TOOL_DIR = Path(__file__).parent
VISUAL_DIR = Path(r"D:\AUTO\VISUAL")
DONE_DIR = Path(r"D:\AUTO\done")
VOICE_DIR = Path(r"D:\AUTO\voice")
PROJECTS_DIR = TOOL_DIR / "PROJECTS"
PROGRESS_FILE = TOOL_DIR / "progress.json"

# Colors - GitHub Dark Theme
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
        self.root.geometry("1000x850")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(True, True)
        self.root.minsize(800, 700)

        # Process handles
        self.srt_process = None
        self.edit_process = None
        self.srt_running = False
        self.edit_running = False

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.create_ui()
        self.refresh_stats()
        self.update_progress()

    def on_closing(self):
        """Handle window close - kill all processes."""
        if self.srt_process:
            try:
                self.srt_process.terminate()
                self.srt_process.wait(timeout=2)
            except:
                try:
                    self.srt_process.kill()
                except:
                    pass

        if self.edit_process:
            try:
                self.edit_process.terminate()
                self.edit_process.wait(timeout=2)
            except:
                try:
                    self.edit_process.kill()
                except:
                    pass

        self.root.destroy()

    def create_ui(self):
        main = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.create_header(main)
        self.create_stats(main)
        self.create_process_cards(main)
        self.create_quick_actions(main)
        self.create_completed_list(main)
        self.create_log_area(main)

    def create_header(self, parent):
        header = tk.Frame(parent, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 20))

        title_frame = tk.Frame(header, bg=COLORS["bg_dark"])
        title_frame.pack(side=tk.LEFT)

        tk.Label(title_frame, text="⚡", font=("Segoe UI", 24),
                bg=COLORS["bg_dark"], fg=COLORS["accent_blue"]).pack(side=tk.LEFT)
        tk.Label(title_frame, text="VE3 Tool", font=("Segoe UI", 20, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(title_frame, text="MASTER", font=("Segoe UI", 10),
                bg=COLORS["bg_dark"], fg=COLORS["accent_green"]).pack(side=tk.LEFT, padx=(10, 0))

        refresh_btn = ModernButton(header, "🔄 Refresh", self.refresh_stats,
                                   COLORS["bg_card_hover"], width=100, height=32)
        refresh_btn.pack(side=tk.RIGHT)

    def create_stats(self, parent):
        stats = tk.Frame(parent, bg=COLORS["bg_dark"])
        stats.pack(fill=tk.X, pady=(0, 20))

        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)
        stats.columnconfigure(2, weight=1)

        self.voice_stat = StatCard(stats, "🎤", "Voice Pending", "0", COLORS["accent_orange"])
        self.voice_stat.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.edit_stat = StatCard(stats, "🎬", "Edit Pending", "0", COLORS["accent_blue"])
        self.edit_stat.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        self.done_stat = StatCard(stats, "✅", "Completed", "0", COLORS["success"])
        self.done_stat.grid(row=0, column=2, sticky="ew")

    def create_process_cards(self, parent):
        processes = tk.Frame(parent, bg=COLORS["bg_dark"])
        processes.pack(fill=tk.X, pady=(0, 20))

        processes.columnconfigure(0, weight=1)
        processes.columnconfigure(1, weight=1)

        self.srt_card = ProcessCard(
            processes, "SRT Generator", "Voice → Subtitle (Whisper AI)",
            "🎤", COLORS["accent_green"],
            self.start_srt, self.stop_srt
        )
        self.srt_card.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.edit_card = ProcessCard(
            processes, "Video Editor", "Image + Voice → MP4",
            "🎬", COLORS["accent_blue"],
            self.start_edit, self.stop_edit
        )
        self.edit_card.grid(row=0, column=1, sticky="ew")

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
            ("⬆ Upload", self.upload_github, COLORS["accent_purple"]),
        ]

        for text, cmd, color in buttons:
            btn = ModernButton(btn_row, text, cmd, color, width=95, height=32)
            btn.pack(side=tk.LEFT, padx=(0, 8))

    def create_completed_list(self, parent):
        """Create list of completed videos."""
        completed_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        completed_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Header
        header = tk.Frame(completed_frame, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 5))

        tk.Label(header, text="🎬 Completed Videos", font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT)

        refresh_btn = tk.Label(header, text="Refresh", font=("Segoe UI", 9),
                              bg=COLORS["bg_dark"], fg=COLORS["accent_blue"], cursor="hand2")
        refresh_btn.pack(side=tk.RIGHT)
        refresh_btn.bind("<Button-1>", lambda e: self.refresh_completed_list())

        # List container
        list_container = tk.Frame(completed_frame, bg=COLORS["border"])
        list_container.pack(fill=tk.BOTH, expand=True)

        # Canvas with scrollbar for the list
        canvas = tk.Canvas(list_container, bg=COLORS["bg_card"], highlightthickness=0, height=150)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)

        self.completed_list_frame = tk.Frame(canvas, bg=COLORS["bg_card"])

        self.completed_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.completed_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.completed_canvas = canvas
        self.refresh_completed_list()

    def refresh_completed_list(self):
        """Refresh the list of completed videos."""
        # Clear existing items
        for widget in self.completed_list_frame.winfo_children():
            widget.destroy()

        if not DONE_DIR.exists():
            tk.Label(self.completed_list_frame, text="No completed videos",
                    bg=COLORS["bg_card"], fg=COLORS["text_dim"],
                    font=("Segoe UI", 9)).pack(pady=10)
            return

        # Get all completed videos sorted by modification time (newest first)
        videos = []
        for folder in DONE_DIR.iterdir():
            if folder.is_dir():
                for mp4 in folder.glob("*.mp4"):
                    videos.append({
                        "path": mp4,
                        "code": folder.name,
                        "mtime": mp4.stat().st_mtime,
                        "size": mp4.stat().st_size
                    })

        videos.sort(key=lambda x: x["mtime"], reverse=True)

        if not videos:
            tk.Label(self.completed_list_frame, text="No completed videos",
                    bg=COLORS["bg_card"], fg=COLORS["text_dim"],
                    font=("Segoe UI", 9)).pack(pady=10)
            return

        # Show last 20 videos
        for vid in videos[:20]:
            self._create_video_item(vid)

    def _create_video_item(self, video_info):
        """Create a clickable video item."""
        item_frame = tk.Frame(self.completed_list_frame, bg=COLORS["bg_card"], cursor="hand2")
        item_frame.pack(fill=tk.X, padx=10, pady=2)

        # Video icon and code
        code_label = tk.Label(item_frame, text=f"▶ {video_info['code']}",
                             font=("Segoe UI", 10), bg=COLORS["bg_card"],
                             fg=COLORS["success"], cursor="hand2")
        code_label.pack(side=tk.LEFT)

        # Size
        size_mb = video_info["size"] / (1024 * 1024)
        size_label = tk.Label(item_frame, text=f"{size_mb:.1f} MB",
                             font=("Segoe UI", 9), bg=COLORS["bg_card"],
                             fg=COLORS["text_dim"])
        size_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Time
        from datetime import datetime as dt
        time_str = dt.fromtimestamp(video_info["mtime"]).strftime("%H:%M")
        time_label = tk.Label(item_frame, text=time_str,
                             font=("Segoe UI", 9), bg=COLORS["bg_card"],
                             fg=COLORS["text_dim"])
        time_label.pack(side=tk.RIGHT)

        # Bind click to play video
        video_path = video_info["path"]
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
        """Read progress file and update Edit card progress bar."""
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

                self.edit_card.set_progress(percent, code, step)

                # Refresh completed list when a video is done
                if step == "Done" and percent == 100:
                    self.root.after(1000, self.refresh_completed_list)
        except:
            pass

        # Refresh every 500ms for smooth updates
        self.root.after(500, self.update_progress)

    def refresh_stats(self):
        voice_count = 0
        if VOICE_DIR.exists():
            for ext in ['.mp3', '.wav', '.m4a']:
                voice_count += len(list(VOICE_DIR.rglob(f"*{ext}")))

        edit_pending = 0
        if VISUAL_DIR.exists():
            for folder in VISUAL_DIR.iterdir():
                if folder.is_dir():
                    has_audio = any(folder.glob("*.mp3")) or any(folder.glob("*.wav"))
                    done_folder = DONE_DIR / folder.name
                    is_done = done_folder.exists() and any(done_folder.glob("*.mp4"))
                    if has_audio and not is_done:
                        edit_pending += 1

        done_count = 0
        if DONE_DIR.exists():
            for folder in DONE_DIR.iterdir():
                if folder.is_dir() and any(folder.glob("*.mp4")):
                    done_count += 1

        self.voice_stat.set_value(voice_count)
        self.edit_stat.set_value(edit_pending)
        self.done_stat.set_value(done_count)

        self.root.after(10000, self.refresh_stats)

    def start_srt(self):
        if self.srt_running:
            return

        self.log("Starting SRT Generator...", "info")
        self.srt_running = True
        self.srt_card.set_running(True)

        def run():
            try:
                # CREATE_NO_WINDOW flag to hide console
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                self.srt_process = subprocess.Popen(
                    [sys.executable, str(TOOL_DIR / "run_srt.py")],
                    cwd=str(TOOL_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW
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
        self.srt_card.set_running(False)
        self.log("SRT Generator stopped", "warning")

    def start_edit(self):
        if self.edit_running:
            return

        self.log("Starting Video Editor...", "info")
        self.edit_running = True
        self.edit_card.set_running(True)

        def run():
            try:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                self.edit_process = subprocess.Popen(
                    [sys.executable, str(TOOL_DIR / "run_edit.py")],
                    cwd=str(TOOL_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW
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
        self.edit_card.set_running(False)
        self.log("Video Editor stopped", "warning")

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
