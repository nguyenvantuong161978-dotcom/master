"""
Test: GPU/CPU slot split for parallel xfade
Kiểm tra:
1. xfade NVENC (slot 0 - GPU) hoạt động đúng
2. xfade libx264 (slot 1 - CPU) hoạt động đúng
3. Cả 2 chạy song song không ảnh hưởng nhau
4. Fallback concat libx264 (thay -c copy) ra đúng duration
"""
import subprocess, tempfile, shutil, threading, time, os
from pathlib import Path

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}" + (f": {detail}" if detail else ""))
        FAIL += 1

def make_clips(tmp, n=10, duration=5.0, codec="libx264"):
    """Tạo n clip test ở 1920x1080."""
    clips = []
    colors = ["red","blue","green","yellow","purple","orange","cyan","white","gray","brown"]
    for i in range(n):
        p = Path(tmp) / f"clip_{i:03d}.mp4"
        color = colors[i % len(colors)]
        if codec == "h264_nvenc":
            enc = ["-c:v","h264_nvenc","-preset","p5","-rc","vbr","-cq","18","-b:v","10M"]
        else:
            enc = ["-c:v","libx264","-preset","fast","-crf","20"]
        cmd = ["ffmpeg","-y","-f","lavfi","-i",
               f"color=c={color}:size=1920x1080:rate=30:duration={duration}",
               *enc, "-pix_fmt","yuv420p","-r","30","-an", str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and p.exists():
            clips.append(p)
    return clips

def get_duration(path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except:
        return 0.0

def run_xfade(clips, out_path, encoder, quality_args):
    """Chạy xfade như production code (batch 15 clips)."""
    td = 0.5
    durations = [get_duration(c) for c in clips]

    inputs = []
    for c in clips:
        inputs.extend(["-i", str(c).replace("\\","/")])

    filter_parts = []
    offset = 0.0
    prev = "[0]"
    for i in range(1, len(clips)):
        offset += durations[i-1] - td
        out_label = f"[v{i}]" if i < len(clips)-1 else "[vout]"
        filter_parts.append(f"{prev}[{i}]xfade=transition=dissolve:duration={td}:offset={offset:.3f}{out_label}")
        prev = out_label

    filter_complex = ";".join(filter_parts) + ";[vout]format=yuv420p[vfinal]"

    cmd = ["ffmpeg","-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map","[vfinal]",
        "-c:v", encoder, *quality_args,
        "-pix_fmt","yuv420p","-r","30",
        str(out_path)
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

def run_fallback_concat(clips, out_path):
    """Fallback concat với libx264 (fix mới, thay -c copy)."""
    tmp = out_path.parent / "concat_list.txt"
    with open(tmp, "w") as f:
        for c in clips:
            f.write(f"file '{str(c).replace(chr(92),'/')}'\n")
    cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",str(tmp),
           "-vf","scale=1920:1080:flags=lanczos,format=yuv420p",
           "-c:v","libx264","-preset","fast","-crf","18",
           "-pix_fmt","yuv420p","-an", str(out_path)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


# ─── TEST 1: xfade GPU (slot 0 - NVENC) ───────────────────────────────────────
print("\n=== TEST 1: xfade GPU slot 0 (NVENC) ===")
t1 = tempfile.mkdtemp()
try:
    clips = make_clips(t1, n=10, duration=5.0, codec="h264_nvenc")
    check("Tạo 10 NVENC clips", len(clips)==10, f"got {len(clips)}")

    out = Path(t1)/"out_gpu.mp4"
    r = run_xfade(clips, out, "h264_nvenc", ["-preset","p5","-rc","vbr","-cq","20","-b:v","15M"])
    check("xfade NVENC rc=0", r.returncode==0, r.stderr[-200:] if r.returncode!=0 else "")

    if out.exists():
        dur = get_duration(out)
        expected = 10*5.0 - 9*0.5  # 10 clips × 5s - 9 transitions × 0.5s = 45.5s
        check(f"xfade NVENC duration ({dur:.1f}s ≈ {expected:.1f}s)", abs(dur-expected)<2.0, f"{dur:.1f} vs {expected:.1f}")
finally:
    shutil.rmtree(t1)


# ─── TEST 2: xfade CPU (slot 1 - libx264) ─────────────────────────────────────
print("\n=== TEST 2: xfade CPU slot 1 (libx264) ===")
t2 = tempfile.mkdtemp()
try:
    clips = make_clips(t2, n=10, duration=5.0, codec="libx264")
    check("Tạo 10 libx264 clips", len(clips)==10)

    out = Path(t2)/"out_cpu.mp4"
    r = run_xfade(clips, out, "libx264", ["-preset","fast","-crf","18","-profile:v","high"])
    check("xfade libx264 rc=0", r.returncode==0, r.stderr[-200:] if r.returncode!=0 else "")

    if out.exists():
        dur = get_duration(out)
        expected = 10*5.0 - 9*0.5
        check(f"xfade libx264 duration ({dur:.1f}s ≈ {expected:.1f}s)", abs(dur-expected)<2.0)
finally:
    shutil.rmtree(t2)


# ─── TEST 3: Parallel GPU + CPU không ảnh hưởng nhau ─────────────────────────
print("\n=== TEST 3: Parallel GPU (slot 0) + CPU (slot 1) đồng thời ===")
t3a = tempfile.mkdtemp()
t3b = tempfile.mkdtemp()
results = {}

def gpu_job():
    clips = make_clips(t3a, n=15, duration=8.0, codec="h264_nvenc")
    out = Path(t3a)/"out.mp4"
    r = run_xfade(clips, out, "h264_nvenc", ["-preset","p5","-rc","vbr","-cq","20","-b:v","15M"])
    results["gpu"] = (r.returncode, get_duration(out) if out.exists() else 0)

def cpu_job():
    clips = make_clips(t3b, n=15, duration=8.0, codec="libx264")
    out = Path(t3b)/"out.mp4"
    r = run_xfade(clips, out, "libx264", ["-preset","fast","-crf","18","-profile:v","high"])
    results["cpu"] = (r.returncode, get_duration(out) if out.exists() else 0)

try:
    t_start = time.time()
    tg = threading.Thread(target=gpu_job)
    tc = threading.Thread(target=cpu_job)
    tg.start(); tc.start()
    tg.join(); tc.join()
    elapsed = time.time() - t_start

    gpu_rc, gpu_dur = results.get("gpu", (-1, 0))
    cpu_rc, cpu_dur = results.get("cpu", (-1, 0))
    expected = 15*8.0 - 14*0.5  # 15 clips × 8s - 14 transitions = 113s

    check("GPU xfade parallel rc=0", gpu_rc==0, f"rc={gpu_rc}")
    check("CPU xfade parallel rc=0", cpu_rc==0, f"rc={cpu_rc}")
    check(f"GPU duration ({gpu_dur:.1f}s ≈ {expected:.1f}s)", abs(gpu_dur-expected)<3.0)
    check(f"CPU duration ({cpu_dur:.1f}s ≈ {expected:.1f}s)", abs(cpu_dur-expected)<3.0)
    print(f"  [INFO] Parallel time: {elapsed:.1f}s")
finally:
    shutil.rmtree(t3a)
    shutil.rmtree(t3b)


# ─── TEST 4: Fallback concat libx264 (thay -c copy) ──────────────────────────
print("\n=== TEST 4: Fallback concat libx264 - đúng duration ===")
t4 = tempfile.mkdtemp()
try:
    # Mix clips: một số NVENC, một số libx264 (simulate mixed codecs)
    nvenc_clips = make_clips(t4, n=5, duration=6.0, codec="h264_nvenc")
    # CPU clips với tên khác trong cùng thư mục
    cpu_dir = Path(t4) / "cpu"
    cpu_dir.mkdir()
    cpu_clips = make_clips(str(cpu_dir), n=5, duration=6.0, codec="libx264")
    all_clips = nvenc_clips + cpu_clips

    out = Path(t4)/"fallback.mp4"
    r = run_fallback_concat(all_clips, out)
    check("Fallback concat rc=0", r.returncode==0, r.stderr[-200:] if r.returncode!=0 else "")

    if out.exists():
        dur = get_duration(out)
        expected = 10*6.0  # 10 clips × 6s, no transitions
        check(f"Fallback duration ({dur:.1f}s ≈ {expected:.1f}s)", abs(dur-expected)<2.0)
finally:
    shutil.rmtree(t4)


# ─── Kết quả ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"KẾT QUẢ: {PASS} PASS / {FAIL} FAIL")
if FAIL == 0:
    print("✓ Tất cả test PASS - OK để chạy production")
else:
    print("✗ Có test FAIL - cần fix trước khi chạy production")
print('='*50)
