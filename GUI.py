#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Tool - Modern GUI Manager
Giao diện quản lý SRT và Edit video - Dark Theme Premium.
"""

import sys
import os
import threading
import subprocess
import time
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

# Colors
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
        """Lighten a hex color."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = min(255, r + 30)
        g = min(255, g + 30)
        b = min(255, b + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw(self, fill_color):
        """Draw the button."""
        self.delete("all")
        r = 8  # Corner radius

        # Draw rounded rectangle
        self.create_arc(0, 0, r*2, r*2, start=90, extent=90, fill=fill_color, outline="")
        self.create_arc(self.width-r*2, 0, self.width, r*2, start=0, extent=90, fill=fill_color, outline="")
        self.create_arc(0, self.height-r*2, r*2, self.height, start=180, extent=90, fill=fill_color, outline="")
        self.create_arc(self.width-r*2, self.height-r*2, self.width, self.height, start=270, extent=90, fill=fill_color, outline="")
        self.create_rectangle(r, 0, self.width-r, self.height, fill=fill_color, outline="")
        self.create_rectangle(0, r, self.width, self.height-r, fill=fill_color, outline="")

        # Draw text
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
        """Enable/disable button."""
        self.disabled = disabled
        self._draw("#4a4a4a" if disabled else self.color)


class StatCard(tk.Frame):
    """Modern stat card with icon."""

    def __init__(self, parent, icon, title, value, color, **kwargs):
        super().__init__(parent, bg=COLORS["bg_card"], **kwargs)

        self.config(highlightbackground=COLORS["border"], highlightthickness=1)

        # Inner padding
        inner = tk.Frame(self, bg=COLORS["bg_card"])
        inner.pack(padx=20, pady=15, fill=tk.BOTH, expand=True)

        # Icon and value row
        top_row = tk.Frame(inner, bg=COLORS["bg_card"])
        top_row.pack(fill=tk.X)

        # Icon
        icon_label = tk.Label(top_row, text=icon, font=("Segoe UI", 24),
                             bg=COLORS["bg_card"], fg=color)
        icon_label.pack(side=tk.LEFT)

        # Value
        self.value_label = tk.Label(top_row, text=value, font=("Segoe UI", 28, "bold"),
                                    bg=COLORS["bg_card"], fg=color)
        self.value_label.pack(side=tk.RIGHT)

        # Title
        title_label = tk.Label(inner, text=title, font=("Segoe UI", 10),
                              bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        title_label.pack(anchor=tk.W, pady=(10, 0))

    def set_value(self, value):
        """Update value."""
        self.value_label.config(text=str(value))


class ProcessCard(tk.Frame):
    """Card for process control."""

    def __init__(self, parent, title, subtitle, icon, color, on_start, on_stop, **kwargs):
        super().__init__(parent, bg=COLORS["bg_card"], **kwargs)

        self.config(highlightbackground=COLORS["border"], highlightthickness=1)
        self.color = color
        self.on_start = on_start
        self.on_stop = on_stop

        # Inner padding
        inner = tk.Frame(self, bg=COLORS["bg_card"])
        inner.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Header row
        header = tk.Frame(inner, bg=COLORS["bg_card"])
        header.pack(fill=tk.X)

        # Icon
        tk.Label(header, text=icon, font=("Segoe UI", 20),
                bg=COLORS["bg_card"], fg=color).pack(side=tk.LEFT)

        # Title
        title_frame = tk.Frame(header, bg=COLORS["bg_card"])
        title_frame.pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(title_frame, text=title, font=("Segoe UI", 12, "bold"),
                bg=COLORS["bg_card"], fg=COLORS["text"]).pack(anchor=tk.W)
        tk.Label(title_frame, text=subtitle, font=("Segoe UI", 9),
                bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(anchor=tk.W)

        # Status indicator
        status_frame = tk.Frame(inner, bg=COLORS["bg_card"])
        status_frame.pack(fill=tk.X, pady=(15, 0))

        self.status_dot = tk.Label(status_frame, text="●", font=("Segoe UI", 12),
                                   bg=COLORS["bg_card"], fg=COLORS["error"])
        self.status_dot.pack(side=tk.LEFT)

        self.status_text = tk.Label(status_frame, text="Stopped", font=("Segoe UI", 10),
                                    bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.status_text.pack(side=tk.LEFT, padx=(5, 0))

        # Buttons
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
        """Update running state."""
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


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class VE3ToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VE3 Tool")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(True, True)

        # Set minimum size
        self.root.minsize(800, 600)

        # Process handles
        self.srt_process = None
        self.edit_process = None
        self.srt_running = False
        self.edit_running = False

        self.create_ui()
        self.refresh_stats()

    def create_ui(self):
        """Create the main UI."""
        # Main container with padding
        main = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        self.create_header(main)

        # Stats row
        self.create_stats(main)

        # Process cards row
        self.create_process_cards(main)

        # Quick actions
        self.create_quick_actions(main)

        # Log area
        self.create_log_area(main)

    def create_header(self, parent):
        """Create header section."""
        header = tk.Frame(parent, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(0, 20))

        # Logo/Title
        title_frame = tk.Frame(header, bg=COLORS["bg_dark"])
        title_frame.pack(side=tk.LEFT)

        tk.Label(title_frame, text="⚡", font=("Segoe UI", 24),
                bg=COLORS["bg_dark"], fg=COLORS["accent_blue"]).pack(side=tk.LEFT)
        tk.Label(title_frame, text="VE3 Tool", font=("Segoe UI", 20, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(title_frame, text="MASTER", font=("Segoe UI", 10),
                bg=COLORS["bg_dark"], fg=COLORS["accent_green"]).pack(side=tk.LEFT, padx=(10, 0))

        # Right side - refresh button
        refresh_btn = ModernButton(header, "🔄 Refresh", self.refresh_stats,
                                   COLORS["bg_card_hover"], width=100, height=32)
        refresh_btn.pack(side=tk.RIGHT)

    def create_stats(self, parent):
        """Create stats cards."""
        stats = tk.Frame(parent, bg=COLORS["bg_dark"])
        stats.pack(fill=tk.X, pady=(0, 20))

        # Configure grid
        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)
        stats.columnconfigure(2, weight=1)

        # Voice pending
        self.voice_stat = StatCard(stats, "🎤", "Voice Pending", "0", COLORS["accent_orange"])
        self.voice_stat.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        # Edit pending
        self.edit_stat = StatCard(stats, "🎬", "Edit Pending", "0", COLORS["accent_blue"])
        self.edit_stat.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        # Done
        self.done_stat = StatCard(stats, "✅", "Completed", "0", COLORS["success"])
        self.done_stat.grid(row=0, column=2, sticky="ew")

    def create_process_cards(self, parent):
        """Create process control cards."""
        processes = tk.Frame(parent, bg=COLORS["bg_dark"])
        processes.pack(fill=tk.X, pady=(0, 20))

        processes.columnconfigure(0, weight=1)
        processes.columnconfigure(1, weight=1)

        # SRT Generator
        self.srt_card = ProcessCard(
            processes, "SRT Generator", "Voice → Subtitle (Whisper AI)",
            "🎤", COLORS["accent_green"],
            self.start_srt, self.stop_srt
        )
        self.srt_card.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        # Video Editor
        self.edit_card = ProcessCard(
            processes, "Video Editor", "Image + Voice → MP4",
            "🎬", COLORS["accent_blue"],
            self.start_edit, self.stop_edit
        )
        self.edit_card.grid(row=0, column=1, sticky="ew")

    def create_quick_actions(self, parent):
        """Create quick action buttons."""
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
            ("⬇ Update", self.update_code, COLORS["accent_orange"]),
        ]

        for text, cmd, color in buttons:
            btn = ModernButton(btn_row, text, cmd, color, width=95, height=32)
            btn.pack(side=tk.LEFT, padx=(0, 8))

    def create_log_area(self, parent):
        """Create log area."""
        log_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        log_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        log_header = tk.Frame(log_frame, bg=COLORS["bg_dark"])
        log_header.pack(fill=tk.X, pady=(0, 5))

        tk.Label(log_header, text="📋 Activity Log", font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_dark"], fg=COLORS["text"]).pack(side=tk.LEFT)

        # Clear button
        clear_btn = tk.Label(log_header, text="Clear", font=("Segoe UI", 9),
                            bg=COLORS["bg_dark"], fg=COLORS["accent_blue"], cursor="hand2")
        clear_btn.pack(side=tk.RIGHT)
        clear_btn.bind("<Button-1>", lambda e: self.clear_log())

        # Log text with custom styling
        log_container = tk.Frame(log_frame, bg=COLORS["border"])
        log_container.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_container, height=10,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            font=("JetBrains Mono", 9),
            relief=tk.FLAT, wrap=tk.WORD,
            insertbackground=COLORS["text"],
            selectbackground=COLORS["accent_blue"],
            padx=10, pady=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Configure tags for colored output
        self.log_text.tag_configure("info", foreground=COLORS["text"])
        self.log_text.tag_configure("success", foreground=COLORS["success"])
        self.log_text.tag_configure("warning", foreground=COLORS["warning"])
        self.log_text.tag_configure("error", foreground=COLORS["error"])
        self.log_text.tag_configure("time", foreground=COLORS["text_dim"])

        # Welcome message
        self.log("VE3 Tool GUI initialized", "success")
        self.log(f"VISUAL: {VISUAL_DIR}")
        self.log(f"DONE: {DONE_DIR}")

    def log(self, message, level="info"):
        """Add message to log with color."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] ", "time")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)

    def clear_log(self):
        """Clear log."""
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared", "info")

    def refresh_stats(self):
        """Refresh statistics."""
        # Count voice files
        voice_count = 0
        if VOICE_DIR.exists():
            for ext in ['.mp3', '.wav', '.m4a']:
                voice_count += len(list(VOICE_DIR.rglob(f"*{ext}")))

        # Count edit pending
        edit_pending = 0
        if VISUAL_DIR.exists():
            for folder in VISUAL_DIR.iterdir():
                if folder.is_dir():
                    has_audio = any(folder.glob("*.mp3")) or any(folder.glob("*.wav"))
                    done_folder = DONE_DIR / folder.name
                    is_done = done_folder.exists() and any(done_folder.glob("*.mp4"))
                    if has_audio and not is_done:
                        edit_pending += 1

        # Count done
        done_count = 0
        if DONE_DIR.exists():
            for folder in DONE_DIR.iterdir():
                if folder.is_dir() and any(folder.glob("*.mp4")):
                    done_count += 1

        # Update cards
        self.voice_stat.set_value(voice_count)
        self.edit_stat.set_value(edit_pending)
        self.done_stat.set_value(done_count)

        # Schedule next refresh
        self.root.after(10000, self.refresh_stats)

    def start_srt(self):
        """Start SRT generator."""
        if self.srt_running:
            return

        self.log("Starting SRT Generator...", "info")
        self.srt_running = True
        self.srt_card.set_running(True)

        def run():
            try:
                self.srt_process = subprocess.Popen(
                    [sys.executable, str(TOOL_DIR / "run_srt.py")],
                    cwd=str(TOOL_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1
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
        """Stop SRT generator."""
        if self.srt_process:
            self.log("Stopping SRT Generator...", "warning")
            self.srt_process.terminate()

    def _on_srt_stopped(self):
        self.srt_running = False
        self.srt_card.set_running(False)
        self.log("SRT Generator stopped", "warning")

    def start_edit(self):
        """Start Video Editor."""
        if self.edit_running:
            return

        self.log("Starting Video Editor...", "info")
        self.edit_running = True
        self.edit_card.set_running(True)

        def run():
            try:
                self.edit_process = subprocess.Popen(
                    [sys.executable, str(TOOL_DIR / "run_edit.py")],
                    cwd=str(TOOL_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1
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
        """Stop Video Editor."""
        if self.edit_process:
            self.log("Stopping Video Editor...", "warning")
            self.edit_process.terminate()

    def _on_edit_stopped(self):
        self.edit_running = False
        self.edit_card.set_running(False)
        self.log("Video Editor stopped", "warning")

    def upload_github(self):
        """Upload to GitHub."""
        self.log("Uploading to GitHub...", "info")

        def do_upload():
            try:
                subprocess.run(["git", "add", "-A"], cwd=str(TOOL_DIR), capture_output=True)
                subprocess.run(["git", "commit", "-m", f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
                              cwd=str(TOOL_DIR), capture_output=True)
                result = subprocess.run(["git", "push", "origin", "main"],
                                        cwd=str(TOOL_DIR), capture_output=True, text=True)

                if result.returncode == 0:
                    self.root.after(0, lambda: self.log("Upload successful!", "success"))
                else:
                    self.root.after(0, lambda: self.log(f"Upload error: {result.stderr}", "error"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Upload error: {e}", "error"))

        threading.Thread(target=do_upload, daemon=True).start()

    def update_code(self):
        """Update code from GitHub."""
        if messagebox.askyesno("Update", "Download latest code from GitHub?\n\nYour config files will be kept."):
            self.log("Updating from GitHub...", "info")
            subprocess.Popen([sys.executable, str(TOOL_DIR / "UPDATE.py")], cwd=str(TOOL_DIR))


# ============================================================================
# MAIN
# ============================================================================

def main():
    root = tk.Tk()

    # Try to set DPI awareness on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = VE3ToolGUI(root)

    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
