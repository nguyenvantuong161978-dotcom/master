"""
Auto Monitor - chay ngam, tu theo doi va fix loi
- Check moi 5 phut
- Detect: pipeline chet, xfade fail, loi pho bien
- Tu restart neu can
- Log vao auto_monitor.log
"""
import subprocess, time, json, os, sys
from pathlib import Path
from datetime import datetime

TOOL_DIR = Path("D:/AUTO/ve3-tool-simple")
RUN_LOG = TOOL_DIR / "run_log.txt"
MON_LOG = TOOL_DIR / "auto_monitor.log"
PYTHON = sys.executable
CHECK_INTERVAL = 1800  # 30 min

def ts():
    return datetime.now().strftime("[%H:%M:%S]")

def log(msg):
    line = f"{ts()} {msg}"
    print(line, flush=True)
    try:
        with open(MON_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass

def get_run_edit_pid():
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "processid,commandline"],
            capture_output=True, text=True, timeout=10
        )
        for line in r.stdout.splitlines():
            if "run_edit.py" in line and "pipeline_manager" not in line and "auto_monitor" not in line:
                for part in reversed(line.strip().split()):
                    if part.isdigit():
                        return int(part)
    except:
        pass
    return None

def get_manager_pid():
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "processid,commandline"],
            capture_output=True, text=True, timeout=10
        )
        for line in r.stdout.splitlines():
            if "pipeline_manager.py" in line:
                for part in reversed(line.strip().split()):
                    if part.isdigit():
                        return int(part)
    except:
        pass
    return None

def get_progress():
    try:
        d = json.loads((TOOL_DIR / "progress.json").read_text(encoding="utf-8"))
        active = [(c, v.get("step","?"), v.get("percent",0))
                  for c, v in d.get("videos", {}).items()
                  if v.get("status") not in (None, "idle", "")]
        return active
    except:
        return []

def get_log_tail(n=200):
    try:
        lines = RUN_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except:
        return ""

def count_visual_projects():
    visual = Path("D:/AUTO/VISUAL")
    try:
        return len([d for d in visual.iterdir() if d.is_dir()])
    except:
        return 0

def restart_pipeline():
    log("Restarting pipeline...")
    # Kill existing
    for _ in range(3):
        pid = get_run_edit_pid()
        if not pid:
            break
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)
            log(f"  Killed PID {pid}")
        except:
            pass
        time.sleep(3)
    # Start new
    log_handle = open(RUN_LOG, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [PYTHON, "-u", "run_edit.py", "--parallel", "2"],
        cwd=str(TOOL_DIR),
        stdout=log_handle,
        stderr=log_handle
    )
    log(f"  Started PID {proc.pid}")
    return proc.pid

def restart_manager():
    log("Restarting manager...")
    pid = get_manager_pid()
    if pid:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)
            log(f"  Killed manager PID {pid}")
        except:
            pass
        time.sleep(2)
    proc = subprocess.Popen(
        [PYTHON, "pipeline_manager.py"],
        creationflags=0x00000008,
        close_fds=True,
        cwd=str(TOOL_DIR)
    )
    log(f"  Manager restarted PID {proc.pid}")

def main():
    log("=" * 50)
    log("Auto Monitor started")
    log("=" * 50)

    last_error_lines = set()
    stall_count = 0
    last_progress_snapshot = {}
    consecutive_dead = 0

    while True:
        try:
            pipeline_pid = get_run_edit_pid()
            manager_pid = get_manager_pid()
            progress = get_progress()
            n_visual = count_visual_projects()
            tail = get_log_tail(100)

            # --- Check 1: Pipeline dead ---
            if not pipeline_pid:
                consecutive_dead += 1
                log(f"WARNING: Pipeline not running (dead count={consecutive_dead}, visual={n_visual})")
                if n_visual > 0 and consecutive_dead >= 2:
                    log("Pipeline dead + projects remain -> restarting")
                    restart_pipeline()
                    consecutive_dead = 0
                    if not manager_pid:
                        restart_manager()
                elif n_visual == 0:
                    log("No projects left. All done!")
                    break
            else:
                consecutive_dead = 0

            # --- Check 2: Manager dead ---
            if not manager_pid and pipeline_pid:
                log("WARNING: Manager dead, restarting...")
                restart_manager()

            # --- Check 3: Stall detection (no progress change in 30 min) ---
            current_snap = {c: p for c, _, p in progress}
            if current_snap and current_snap == last_progress_snapshot:
                stall_count += 1
                if stall_count >= 4:  # 4 x 30min = 2h no change
                    log(f"STALL detected (2h no progress change): {current_snap}")
                    log("Restarting pipeline to recover...")
                    restart_pipeline()
                    stall_count = 0
            else:
                stall_count = 0
                last_progress_snapshot = current_snap

            # --- Check 4: Log errors ---
            new_errors = []
            for line in tail.splitlines():
                if any(x in line for x in ["[ERROR]", "Traceback", "rc=-22", "UnicodeEncodeError"]):
                    if line not in last_error_lines:
                        new_errors.append(line)
                        last_error_lines.add(line)
            if new_errors:
                log(f"ERRORS in log ({len(new_errors)} new):")
                for e in new_errors[-5:]:
                    log(f"  {e.strip()}")

            # --- Check 5: Duplicate pipelines ---
            try:
                r = subprocess.run(
                    ["wmic", "process", "where", "name='python.exe'",
                     "get", "processid,commandline"],
                    capture_output=True, text=True, timeout=10
                )
                pids = []
                for line in r.stdout.splitlines():
                    if "run_edit.py" in line and "pipeline_manager" not in line and "auto_monitor" not in line:
                        for part in reversed(line.strip().split()):
                            if part.isdigit():
                                pids.append(int(part))
                                break
                if len(pids) > 1:
                    log(f"WARNING: {len(pids)} pipeline processes! Killing extras: {pids[1:]}")
                    for pid in pids[1:]:
                        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)
            except:
                pass

            # --- Status log every 30 min ---
            now_min = datetime.now().minute
            if now_min % 30 < 5:  # log at :00 and :30
                prog_str = " | ".join(f"{c}:{s}({p}%)" for c,s,p in progress) if progress else "idle"
                log(f"STATUS: pipeline={pipeline_pid} | visual={n_visual} | {prog_str}")

        except Exception as e:
            log(f"Monitor error: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
