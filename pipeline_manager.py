"""
Pipeline Manager v3 - robust keepalive + smart restart
- Detects existing run_edit.py and adopts or waits
- Restarts process when it exits
- Triggers code-update restart when specific videos complete
"""
import subprocess
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime

TOOL_DIR = Path("D:/AUTO/ve3-tool-simple")
LOG_FILE = TOOL_DIR / "run_log.txt"
MANAGER_LOG = TOOL_DIR / "manager.log"
PYTHON = sys.executable
PARALLEL = 2

def ts():
    return datetime.now().strftime("[%H:%M:%S]")

def mlog(msg, level="INFO"):
    line = f"{ts()} [{level}] {msg}"
    print(line, flush=True)
    try:
        with open(MANAGER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def find_run_edit_pid():
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "processid,commandline"],
            capture_output=True, text=True, timeout=10
        )
        for line in r.stdout.splitlines():
            if "run_edit.py" in line and "pipeline_manager" not in line:
                for part in reversed(line.strip().split()):
                    if part.isdigit():
                        return int(part)
    except Exception as e:
        mlog(f"PID search error: {e}", "WARN")
    return None

def kill_run_edit():
    for _ in range(5):
        pid = find_run_edit_pid()
        if not pid:
            mlog("No run_edit.py process found (already dead)")
            return
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)
            mlog(f"Killed PID {pid}")
        except Exception as e:
            mlog(f"Kill error: {e}", "WARN")
        time.sleep(3)

def start_pipeline():
    mlog(f"Starting: python -u run_edit.py --parallel {PARALLEL}")
    try:
        log_handle = open(LOG_FILE, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [PYTHON, "-u", "run_edit.py", "--parallel", str(PARALLEL)],
            cwd=str(TOOL_DIR),
            stdout=log_handle,
            stderr=log_handle
        )
        mlog(f"Started PID {proc.pid}")
        return proc, log_handle
    except Exception as e:
        mlog(f"Start failed: {e}", "ERROR")
        return None, None

def log_contains(text):
    try:
        return text in LOG_FILE.read_text(encoding="utf-8", errors="ignore")
    except:
        return False

def scan_visual_projects():
    visual = Path("D:/AUTO/VISUAL")
    if not visual.exists():
        return []
    return [d.name for d in visual.iterdir() if d.is_dir()]

def get_progress_summary():
    try:
        prog = json.loads((TOOL_DIR / "progress.json").read_text(encoding="utf-8"))
        videos = prog.get("videos", {})
        active = [(c, v.get("step","?"), v.get("percent",0))
                  for c, v in videos.items() if v.get("status") != "idle"]
        return " | ".join(f"{c}:{s}({p}%)" for c,s,p in active[:3])
    except:
        return ""

def main():
    mlog("=" * 60)
    mlog("Pipeline Manager v4 started - keepalive only")
    mlog("=" * 60)

    current_proc = None
    current_log_handle = None
    consecutive_fails = 0
    MAX_FAILS = 5
    check_count = 0

    # Adopt existing run_edit.py process if already running (e.g. manually started)
    existing_pid = find_run_edit_pid()
    if existing_pid:
        mlog(f"Adopting existing run_edit.py PID={existing_pid} (started externally)")
        # We can't adopt a Popen object, so we just wait for it to die
        import psutil
        try:
            adopted = psutil.Process(existing_pid)
            mlog(f"Waiting for PID {existing_pid} to finish before taking over...")
            adopted.wait()
            mlog(f"Adopted process PID={existing_pid} exited. Starting keepalive...")
        except Exception:
            mlog("psutil not available or process gone, proceeding to keepalive loop")

    try:
        while True:
            check_count += 1
            projects = scan_visual_projects()

            # First check if there's already an external run_edit.py we don't own
            if current_proc is None:
                ext_pid = find_run_edit_pid()
                if ext_pid:
                    mlog(f"External run_edit.py running (PID={ext_pid}), waiting...")
                    time.sleep(120)
                    continue

            if current_proc is None or current_proc.poll() is not None:
                rc = current_proc.poll() if current_proc else None

                if current_proc is not None:
                    if rc == 0:
                        mlog("Process exited cleanly.")
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1
                        mlog(f"Process died rc={rc}. Fails={consecutive_fails}/{MAX_FAILS}", "WARN")
                    if current_log_handle:
                        try: current_log_handle.close()
                        except: pass
                    if consecutive_fails >= MAX_FAILS:
                        mlog("Too many fails. Stopping.", "ERROR")
                        break

                # Before starting new pipeline, check for external instance (avoid duplicate)
                ext_pid = find_run_edit_pid()
                if ext_pid:
                    mlog(f"External run_edit.py PID={ext_pid} already running, not spawning duplicate")
                    current_proc = None
                    time.sleep(120)
                    continue

                if projects:
                    mlog(f"{len(projects)} projects: {', '.join(sorted(projects)[:6])}")
                    current_proc, current_log_handle = start_pipeline()
                    if not current_proc:
                        consecutive_fails += 1
                        time.sleep(30)
                        continue
                else:
                    mlog("VISUAL empty! All done. Sleeping 5 min then recheck...")
                    time.sleep(300)
                    if not scan_visual_projects():
                        mlog("Still empty. Manager exiting.")
                        break
            else:
                # Process running
                if check_count % 5 == 0:  # Log every 5 checks = every 10 min
                    prog = get_progress_summary()
                    mlog(f"OK PID={current_proc.pid} | {len(projects)} in VISUAL | {prog}")

            time.sleep(120)  # Check every 2 min

    except KeyboardInterrupt:
        mlog("Interrupted by user")
    finally:
        if current_log_handle:
            try: current_log_handle.close()
            except: pass
        mlog("Manager exiting.")

if __name__ == "__main__":
    main()
