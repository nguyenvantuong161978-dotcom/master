"""Monitor chạy nền: check mỗi 15 phút, ghi log, tự fix nếu cần."""
import subprocess, time, json, os, sys
from pathlib import Path
from datetime import datetime

TOOL_DIR = Path("D:/AUTO/ve3-tool-simple")
LOG = TOOL_DIR / "monitor_session.log"
RUN_LOG = TOOL_DIR / "run_log.txt"
PYTHON = sys.executable
CHECK_INTERVAL = 900  # 15 phút

def ts():
    return datetime.now().strftime("[%H:%M:%S]")

def mlog(msg):
    line = f"{ts()} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_pipeline_pid():
    try:
        r = subprocess.run(["wmic","process","where","name='python.exe'",
                            "get","processid,commandline"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if "run_edit.py" in line and "pipeline_manager" not in line and "auto_monitor" not in line and "monitor_session" not in line:
                for part in reversed(line.strip().split()):
                    if part.isdigit():
                        return int(part)
    except: pass
    return None

def get_progress():
    try:
        d = json.loads((TOOL_DIR/"progress.json").read_text(encoding="utf-8"))
        return {c: (v.get("step","?"), v.get("percent",0))
                for c,v in d.get("videos",{}).items()
                if v.get("status") not in (None,"idle","")}
    except: return {}

def count_visual():
    try: return len([d for d in Path("D:/AUTO/VISUAL").iterdir() if d.is_dir()])
    except: return 0

def count_done():
    try: return len([d for d in Path("D:/AUTO/done").iterdir() if d.is_dir()])
    except: return 0

def restart_pipeline():
    # Kill existing
    for _ in range(3):
        pid = get_pipeline_pid()
        if not pid: break
        subprocess.run(["taskkill","/PID",str(pid),"/F"], capture_output=True)
        mlog(f"  Killed PID {pid}")
        time.sleep(3)
    # Start new
    lh = open(str(RUN_LOG), "a", encoding="utf-8")
    proc = subprocess.Popen([PYTHON,"-u","run_edit.py","--parallel","2"],
                            cwd=str(TOOL_DIR), stdout=lh, stderr=lh)
    mlog(f"  Restarted PID={proc.pid}")

def get_recent_errors():
    try:
        lines = RUN_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
        errors = [l for l in lines if "[ERROR]" in l or "xfade failed" in l or "Concat too short" in l]
        return errors[-5:]
    except: return []

mlog("="*50)
mlog("Monitor session started (check every 15 min)")
mlog("="*50)

stall_snap = {}
stall_count = 0
last_done_count = count_done()

while True:
    try:
        pid = get_pipeline_pid()
        progress = get_progress()
        n_visual = count_visual()
        n_done = count_done()

        # Kiểm tra pipeline chết
        if not pid:
            if n_visual > 0:
                mlog(f"WARNING: Pipeline chết! visual={n_visual} → restart")
                restart_pipeline()
                stall_count = 0
            else:
                mlog("Tất cả video done, kết thúc monitor.")
                break
        else:
            # Kiểm tra stall (không đổi progress sau 30 phút = 2 lần check)
            snap = {c: p for c,(s,p) in progress.items()}
            if snap and snap == stall_snap:
                stall_count += 1
                if stall_count >= 4:  # 4×15 = 60 phút không đổi
                    mlog(f"STALL 60 phút! {snap} → restart")
                    restart_pipeline()
                    stall_count = 0
            else:
                stall_count = 0
                stall_snap = snap

        # Log tiến độ
        new_done = n_done - last_done_count
        prog_str = " | ".join(f"{c}:{s}({p}%)" for c,(s,p) in progress.items()) or "idle"
        mlog(f"STATUS pid={pid} | queue={n_visual} | done={n_done}(+{new_done}) | {prog_str}")
        last_done_count = n_done

        # Log errors mới
        errors = get_recent_errors()
        for e in errors:
            mlog(f"  ERR: {e.strip()[-120:]}")

    except Exception as e:
        mlog(f"Monitor error: {e}")

    time.sleep(CHECK_INTERVAL)
