#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Master Launcher - 3 panels in 1 window
Tab 1: VE3 Edit (GUI.pyw)
Tab 2: VM Make (master_contro.py)
Tab 3: VM Upload (upload/control.py)
"""

import sys
import os
import tkinter as tk
from pathlib import Path

TOOL_DIR = Path(__file__).parent.resolve()

# Add paths for imports
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(Path(r"D:\upload")))


# ============================================================================
# COLORS
# ============================================================================

BG = "#0d0d14"
BG_TAB = "#151520"
BG_ACTIVE = "#1a1a2e"
FG = "#e0e0e0"
FG_DIM = "#5a5f75"

TAB_COLORS = {
    "ve3":    "#60a5fa",  # blue
    "vm":     "#a78bfa",  # purple
    "upload": "#fb923c",  # orange
}


# ============================================================================
# LAUNCHER
# ============================================================================

class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VE3 Master")
        self.root.geometry("1200x900")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(900, 600)

        self.tabs = {}
        self.tab_frames = {}
        self.tab_apps = {}
        self.current_tab = None

        self._build_tab_bar()
        self._build_content()

        # Start with VE3 Edit tab
        self._switch_tab("ve3")

        # Handle close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_tab_bar(self):
        bar = tk.Frame(self.root, bg=BG, pady=2)
        bar.pack(fill=tk.X)

        tab_defs = [
            ("ve3",    "VE3 EDIT"),
            ("vm",     "VM LAM VIDEO"),
            ("upload", "VM UPLOAD"),
        ]

        for key, label in tab_defs:
            color = TAB_COLORS[key]
            btn = tk.Label(bar, text=f"  {label}  ", font=("Segoe UI", 11, "bold"),
                          bg=BG_TAB, fg=FG_DIM, padx=15, pady=6, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=(2, 0))
            btn.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))
            self.tabs[key] = {"btn": btn, "color": color}

    def _build_content(self):
        self.content = tk.Frame(self.root, bg=BG)
        self.content.pack(fill=tk.BOTH, expand=True)

        # Pre-create frames for each tab
        for key in ["ve3", "vm", "upload"]:
            frame = tk.Frame(self.content, bg=BG)
            self.tab_frames[key] = frame

    def _switch_tab(self, key):
        if self.current_tab == key:
            return

        # Update tab button styles
        for k, tab in self.tabs.items():
            if k == key:
                tab["btn"].config(bg=BG_ACTIVE, fg=tab["color"])
            else:
                tab["btn"].config(bg=BG_TAB, fg=FG_DIM)

        # Hide current frame
        if self.current_tab:
            self.tab_frames[self.current_tab].pack_forget()

        # Show target frame
        self.tab_frames[key].pack(fill=tk.BOTH, expand=True)
        self.current_tab = key

        # Lazy-load app on first switch
        if key not in self.tab_apps:
            self._load_tab(key)

    def _load_tab(self, key):
        frame = self.tab_frames[key]

        try:
            if key == "ve3":
                # Import and create VE3 Edit GUI
                loading = tk.Label(frame, text="Dang tai VE3 Edit...",
                                  font=("Segoe UI", 12), bg=BG, fg=FG_DIM)
                loading.pack(pady=50)
                self.root.update()

                import importlib.util
                spec = importlib.util.spec_from_file_location("GUI_pyw", str(TOOL_DIR / "GUI.pyw"))
                gui_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_mod)
                VE3ToolGUI = gui_mod.VE3ToolGUI
                loading.destroy()
                app = VE3ToolGUI(self.root, parent=frame)
                self.tab_apps[key] = app

            elif key == "vm":
                loading = tk.Label(frame, text="Dang tai VM Control...",
                                  font=("Segoe UI", 12), bg=BG, fg=FG_DIM)
                loading.pack(pady=50)
                self.root.update()

                sys.path.insert(0, str(TOOL_DIR / "control"))
                from master_contro import MasterControlGUI
                loading.destroy()
                app = MasterControlGUI(self.root, parent=frame)
                self.tab_apps[key] = app

            elif key == "upload":
                loading = tk.Label(frame, text="Dang tai Upload Control...",
                                  font=("Segoe UI", 12), bg=BG, fg=FG_DIM)
                loading.pack(pady=50)
                self.root.update()

                from control import App as UploadApp
                loading.destroy()
                app = UploadApp(root=self.root, parent=frame)
                self.tab_apps[key] = app

        except Exception as e:
            import traceback
            for w in frame.winfo_children():
                w.destroy()
            tk.Label(frame, text=f"Loi tai module: {e}",
                    font=("Consolas", 10), bg=BG, fg="#f87171",
                    wraplength=800, justify=tk.LEFT).pack(pady=20, padx=20)
            tk.Label(frame, text=traceback.format_exc(),
                    font=("Consolas", 8), bg=BG, fg=FG_DIM,
                    wraplength=800, justify=tk.LEFT).pack(padx=20)

    def _on_close(self):
        # Stop VE3 processes if running
        ve3 = self.tab_apps.get("ve3")
        if ve3:
            try:
                ve3.on_closing()
                return  # on_closing calls root.destroy()
            except Exception:
                pass

        # Stop master control
        vm = self.tab_apps.get("vm")
        if vm and hasattr(vm, '_stop'):
            vm._stop.set()

        self.root.destroy()

    def run(self):
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_height()) // 2
        self.root.geometry(f"+{x}+{y}")
        self.root.mainloop()


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = Launcher()
    app.run()


if __name__ == "__main__":
    main()
