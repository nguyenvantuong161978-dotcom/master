# thumbnail_generator.py – QUY HOẠCH FULL
from PIL import ImageFilter
import numpy as np
import io, os

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import unicodedata
import re

# ========== CONFIG ==========
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR.parent / "config" / "config.json"
ASSETS_DIR = BASE_DIR / "assets"
BG_DIR = BASE_DIR / "pic"
OUTPUT_DIR = BASE_DIR / "thumbnails"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

layout = {
    "img_width": 1920,
    "img_height": 1080,
    "right_prop": 0.35,
    "left_margin_text": 30,
    "margin_x": 50,
    "bar_padding": 20,
    "top_padding": 5,
    "bottom_padding": 5,
    "line_spacing": 10,
    "fade_width": 40,
    "min_font": 36,
    "max_font": 388,
    "keyword_min_font": 28,
    "keyword_max_font": 200,
    "min_bar_frac": 1 / 6,
    "max_bar_frac": 1 / 6,
    "text_stroke": 6,                 # độ dày viền chữ text thumb
    "text_stroke_color": "#000000",   # màu viền chữ text thumb
    # 🎨 Màu sắc theo yêu cầu
    "text_bg_color": "#fcdb39",
    "text_color": "#ffffff",
    "highlight_color": "#fff500",
    "keyword_color": "#c50820",
    "keyword_stroke": 0,
    "keyword_stroke_color": "#000000",

    # 🔧 TẮT nền hook bar
    "use_hook_bar": False,
    # (giữ "bar_color" nếu muốn bật lại sau này, nhưng không bắt buộc)
    # "bar_color": "#ff0000",
}


FONT_PATH = BASE_DIR.parent / "fonts" / "00034-UTM-AvoBold.ttf"
KEY_FONT_PATH = BASE_DIR.parent / "fonts" / "Anton-Regular.ttf"

# ========== TEXT UTILS ==========

def _to_rgb_flat(im: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """Chuyển ảnh (có alpha) về RGB trên nền bg (trắng)."""
    if im.mode in ("RGBA", "LA"):
        rgb = Image.new("RGB", im.size, bg)
        alpha = im.split()[-1] if im.mode in ("RGBA", "LA") else None
        rgb.paste(im.convert("RGBA"), mask=alpha)
        return rgb
    return im.convert("RGB")

def save_under_2mb_jpeg(im: Image.Image, out_path: Path, max_bytes: int = 2_000_000) -> None:
    """
    Nén JPEG progressive + optimize sao cho file ≤ max_bytes.
    - Giảm chất lượng dần (95 → 50).
    - Nếu vẫn lớn, thu nhỏ 90% và thử lại (tối đa vài lần).
    """
    im_rgb = _to_rgb_flat(im)

    # Thử nhiều vòng: giảm chất lượng, nếu không đủ thì giảm kích thước rồi thử lại
    for _shrink_round in range(4):  # tối đa 4 lần thu nhỏ (rất hiếm khi cần)
        for q in range(95, 49, -5):  # 95, 90, 85, ..., 50
            buf = io.BytesIO()
            im_rgb.save(
                buf,
                format="JPEG",
                quality=q,
                optimize=True,
                progressive=True,
                subsampling="4:2:0",
            )
            size = buf.tell()
            if size <= max_bytes:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(buf.getvalue())
                return
        # Nếu vẫn > max_bytes, thu nhỏ ảnh 90% và thử lại
        new_w = int(im_rgb.width * 0.9)
        new_h = int(im_rgb.height * 0.9)
        if new_w < 640 or new_h < 360:  # đừng co quá nhỏ cho thumbnail 16:9
            break
        im_rgb = im_rgb.resize((new_w, new_h), Image.LANCZOS)

    # Nếu mọi cách đều không đạt (rất hiếm), vẫn lưu chất lượng 50 để đảm bảo có file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im_rgb.save(
        out_path,
        format="JPEG",
        quality=50,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )





# ========= MATTE HELPERS =========
try:
    import torch
    from torchvision import transforms
    TORCH_OK = True
except:
    TORCH_OK = False
try:
    import cv2
    CV2_OK = True
except:
    CV2_OK = False
try:
    from rembg import remove as rembg_remove
    REMBG_OK = True
except:
    REMBG_OK = False

_MODNET = {"model": None, "device": None}

def _load_modnet(weight_path: Path = Path("MODNet/modnet_photographic_portrait_matting.pth")) -> bool:
    if not TORCH_OK: return False
    if _MODNET["model"]: return True
    try:
        import sys
        modnet_src = Path("MODNet") / "src"
        if modnet_src.exists(): sys.path.append(str(modnet_src))
        from models.modnet import MODNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MODNet(backbone_pretrained=False)
        sd = torch.load(str(weight_path), map_location=device)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.to(device).eval()
        _MODNET.update({"model": model, "device": device})
        return True
    except: return False

def _preprocess_512(img):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return tfm(img.convert("RGB").resize((512, 512))).unsqueeze(0)

def smart_cutout(pil_image, modnet_weight=Path("MODNet/modnet_photographic_portrait_matting.pth"), upscale=6, alpha_blur=3):
    im = pil_image
    if im.mode in ("RGBA", "LA") and max(im.split()[-1].getextrema()) <= 250:
        return im.convert("RGBA")
    if _load_modnet(modnet_weight) and CV2_OK:
        with torch.no_grad():
            model, device = _MODNET["model"], _MODNET["device"]
            w, h = im.size
            big = im.convert("RGB").resize((w*upscale, h*upscale), Image.LANCZOS)
            tin = _preprocess_512(big).to(device)
            _, _, matte = model(tin, True)
            matte_np = matte[0][0].detach().cpu().numpy()
            matte_np = cv2.resize(matte_np, big.size, interpolation=cv2.INTER_AREA)
            alpha = (np.clip(matte_np, 0, 1) * 255).astype(np.uint8)
            rgba = Image.fromarray(np.dstack((np.array(big), alpha)), "RGBA")
            a = rgba.split()[-1].filter(ImageFilter.GaussianBlur(alpha_blur))
            rgba = Image.merge("RGBA", (*rgba.split()[:3], a))
            return rgba.resize((w, h), Image.LANCZOS)
    if REMBG_OK:
        out = rembg_remove(im.convert("RGB"))
        return (out if isinstance(out, Image.Image) else Image.open(out)).convert("RGBA")
    return im.convert("RGBA")

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip().replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

def wrap_lines(draw, text, font, max_w):
    words = text.replace("\n", " ").split(" ")
    line, lines = "", []
    for w in words:
        test = f"{line} {w}".strip()
        if draw.textlength(test, font=font) <= max_w:
            line = test
        else:
            if line == "":
                lines.append(w)
            else:
                lines.append(line)
                line = w
    if line:
        lines.append(line)
    return lines

def wrap_balanced(draw, text, font, max_w, max_lines=2):
    words = text.split()
    if draw.textlength(text, font=font) <= max_w:
        return [text]
    best, diff_best = None, float("inf")
    for i in range(1, len(words)):
        lines = [" ".join(words[:i]), " ".join(words[i:])]
        if len(lines) > max_lines:
            continue
        widths = [draw.textlength(l, font=font) for l in lines]
        if all(w <= max_w for w in widths):
            diff = abs(widths[0] - widths[1])
            if diff < diff_best:
                best, diff_best = lines, diff
    return best if best else [" ".join(words)]
# (Phần 2 tiếp theo – Font fitting, highlight & keyword)

def auto_fit(draw, text, font_path, max_w, max_h, min_size, max_size):
    best_fit = None
    for size in range(max_size, min_size - 1, -1):  # từ to → nhỏ
        font = ImageFont.truetype(str(font_path), size)
        lines = wrap_lines(draw, text, font, max_w)
        total_height = len(lines) * size + (len(lines) - 1) * layout["line_spacing"]
        if total_height <= max_h and all(draw.textlength(l, font=font) <= max_w for l in lines):
            best_fit = (lines, font)
            break  # chọn ngay kích thước lớn nhất vừa
    if best_fit:
        return best_fit
    # fallback nếu không dòng nào vừa
    font = ImageFont.truetype(str(font_path), min_size)
    return wrap_lines(draw, text, font, max_w), font

def fit_bar_text(draw, text, max_w):
    txt = text.upper().replace("\n", " ").strip()
    for sz in range(layout["keyword_max_font"], layout["keyword_min_font"] - 1, -2):
        f = ImageFont.truetype(str(KEY_FONT_PATH), sz)
        if draw.textlength(txt, font=f) <= max_w:
            return [txt], f
    f = ImageFont.truetype(str(KEY_FONT_PATH), layout["keyword_min_font"])
    return [txt], f

def build_flex_regex(phrase):
    tokens = [re.escape(t) for t in phrase.strip().split()]
    if not tokens:
        return re.compile(r"$^")
    first = tokens[0]
    if first[0].isalnum():
        first = r"\b" + first
    last = tokens[-1]
    if last[-1].isalnum():
        last = last + r"\b"
    MID = r"[^\w]*\s*"
    core = MID.join([first, *tokens[1:-1], last] if len(tokens) > 1 else [first])
    return re.compile(core, re.I | re.S)

def split_phrases(raw):
    return [p.strip() for p in str(raw).split("|") if p.strip()]

def merge_ranges(ranges):
    if not ranges:
        return []
    ranges.sort()
    merged = [list(ranges[0])]
    for st, ed in ranges[1:]:
        if st <= merged[-1][1]:
            merged[-1][1] = max(ed, merged[-1][1])
        else:
            merged.append([st, ed])
    return [tuple(r) for r in merged]

def mark_ranges(line, glb_pos, spans):
    if not spans:
        return [(line, False)]
    segs, cur = [], 0
    for st, ed in spans:
        if ed <= glb_pos or st >= glb_pos + len(line):
            continue
        s = max(st - glb_pos, 0)
        e = min(ed - glb_pos, len(line))
        if s > cur:
            segs.append((line[cur:s], False))
        segs.append((line[s:e], True))
        cur = e
    if cur < len(line):
        segs.append((line[cur:], False))
    return segs

def draw_rich_line(draw, x, y, segments, font):
    stroke_w = layout.get("text_stroke", 0)
    stroke_col = layout.get("text_stroke_color", "#000000")
    for seg, is_high in segments:
        color = layout["highlight_color"] if is_high else layout["text_color"]
        draw.text(
            (x, y), seg, fill=color, font=font,
            stroke_width=stroke_w, stroke_fill=stroke_col
        )
        x += draw.textlength(seg, font=font)


def make_fade(w, h, color=(0,0,0,255)):
    strip = Image.new("RGBA", (w, h), color)
    grad  = Image.new("L", (w, 1))
    for x in range(w):
        grad.putpixel((x,0), int(255*(1-x/w)))
    strip.putalpha(grad.resize((w, h)))
    return strip

def make_grad_bar(w, h, rgb=(255,214,0)):
    bar = Image.new("RGBA", (w, h), (*rgb, 255))
    alpha = Image.new("L", (w, 1))
    for x in range(w):
        alpha.putpixel((x,0), int(255*(1-x/w)))
    bar.putalpha(alpha.resize((w, h)))
    return bar
# (Phần 3 – Xử lý ảnh nền, tạo thumbnail & main loop)

def resolve_bg(code):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = BG_DIR / f"{code}{ext}"
        if path.exists():
            return path
    return None

def generate(row):
    img_w, img_h = layout["img_width"], layout["img_height"]
    right_w = int(img_w * layout["right_prop"])
    panel_w = img_w - right_w

    out_path = OUTPUT_DIR / f"{row.image_code}.jpg"

    if out_path.exists():
        print(f"⏩ {out_path.name} đã tồn tại – skip")
        return

    bg_path = resolve_bg(row.image_code)
    if not bg_path:
        print(f"⚠️ {row.image_code}: Không tìm thấy ảnh nền trong 'pic/' – skip")
        return

    # --- Canvas & ảnh bên phải (giống KA2) ---
    canvas = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))
    src = Image.open(bg_path)
    cut = smart_cutout(src)
    if cut.mode == "RGBA":
        alpha = cut.split()[-1]
        bbox = alpha.getbbox()
        if bbox: cut = cut.crop(bbox)
    # Scale theo chiều cao để nhân vật to như KA2
    orig_w, orig_h = cut.size
    scale = img_h / orig_h
    new_w = int(orig_w * scale)
    new_h = img_h
    cut = cut.resize((new_w, new_h), Image.LANCZOS)
    orig_w, orig_h = cut.size

    # Center ngang trong panel phải
    px = panel_w + (right_w - orig_w) // 2
    py = 0  # Top-aligned vì đã scale full height

    # Nếu nhân vật rộng hơn panel, crop center để vừa
    if orig_w > right_w:
        crop_left = (orig_w - right_w) // 2
        cut = cut.crop((crop_left, 0, crop_left + right_w, orig_h))
        px = panel_w

    canvas.paste(cut, (px, py), cut)

    draw = ImageDraw.Draw(canvas)

    # --- Panel trái (text thumb) & hook bar ---
    # --- Panel trái (text thumb) & hook bar ---
    bar_h   = img_h // 6              # 1/6 ảnh = hook
    text_h  = img_h - bar_h           # 5/6 ảnh = vùng text
    bar_top = text_h                  # hook bắt đầu sau vùng text

    # Nền panel trái: phủ đủ (cover) và căn trái
    text_bg_img_path = ASSETS_DIR / "KA1-T4.jpg"
    bg_img = Image.open(text_bg_img_path).convert("RGB")

    # fit theo kích thước panel_w x img_h, phủ full chiều cao panel trái
    bg_fit = ImageOps.fit(
        bg_img,
        (panel_w, img_h),
        method=Image.LANCZOS,
        centering=(0.0, 0.5)  # bám trái, crop bên phải nếu thừa
    )
    canvas.paste(bg_fit, (0, 0))


    # --- Text content ---
    text = clean_text(row.text_thumb)
    phrases = [clean_text(p) for p in split_phrases(row.highlights)]

    max_w_txt = panel_w - layout["margin_x"] - layout["left_margin_text"]
    text_top = layout["top_padding"]

    # ✅ LỚP AN TOÀN #1: chừa đệm tính theo chiều cao bar (vd 30%)
    SAFE_GAP_FRAC = 0.30
    safe_gap_px = int(bar_h * SAFE_GAP_FRAC)

    # Vùng cao thực sự dành cho text
    max_h_txt = text_h - text_top - layout["bottom_padding"] - safe_gap_px
    if max_h_txt <= 0:
        max_h_txt = max(1, text_h // 3)  # fallback đề phòng cấu hình quá sát

    # Fit text vào vùng 5/6
    lines, font = auto_fit(
        draw,
        text,
        FONT_PATH,
        max_w_txt,
        max_h_txt,
        layout["min_font"],
        layout["max_font"]
    )



    dyn_sp = layout["line_spacing"]
    char_h_real = len(lines) * font.size + (len(lines) - 1) * dyn_sp
    # canh giữa theo trục dọc trong vùng text
    y = text_top + max(0, (max_h_txt - char_h_real) // 2)


    # Highlight theo cụm
    big = "\n".join(lines)
    spans = []
    for ph in phrases:
        for m in build_flex_regex(ph).finditer(big):
            spans.append((m.start(), m.end()))
    spans = merge_ranges(spans)

    glb = 0
    for ln in lines:
        segs = mark_ranges(ln, glb, spans)
        w_line = sum(draw.textlength(seg, font=font) for seg, _ in segs)
        x = layout["left_margin_text"] + (max_w_txt - w_line) // 2
        draw_rich_line(draw, x, y, segs, font)
        y += font.size + dyn_sp
        glb += len(ln) + 1


    # --- Hook bar (1/6) ---
    if layout.get("use_hook_bar", True):
        bar_color = layout.get("bar_color", "#ff0000")
        draw.rectangle([(0, bar_top), (panel_w, img_h)], fill=bar_color)
    # Nếu use_hook_bar=False → không vẽ gì (nền trong suốt)



    kw_text = clean_text(row.hook).upper()
    max_w_bar = panel_w - 2 * layout["bar_padding"]

    # giữ khoảng trống trên/dưới trong bar
    safe_ratio = 0.7
    max_h_bar = int(bar_h * safe_ratio)

    # Fit chữ hook vào bar
    best_font = None
    for sz in range(layout["keyword_max_font"], layout["keyword_min_font"] - 1, -2):
        f = ImageFont.truetype(str(KEY_FONT_PATH), sz)
        if draw.textlength(kw_text, font=f) <= max_w_bar:
            bbox = f.getbbox(kw_text)
            if (bbox[3] - bbox[1]) <= max_h_bar:
                best_font = f
                break

    if best_font:
        kw_font = best_font
        kw_x = panel_w // 2
        kw_y = bar_top + bar_h // 2
        draw.text(
            (kw_x, kw_y),
            kw_text,
            fill=layout["keyword_color"],
            font=kw_font,
            anchor="mm",
            stroke_width=layout["keyword_stroke"],
            stroke_fill=layout["keyword_stroke_color"]
        )

    save_under_2mb_jpeg(canvas, out_path, max_bytes=2_000_000)
    final_size = out_path.stat().st_size if out_path.exists() else 0
    print(f"✅ Saved: {out_path.name} ({final_size/1024:.1f} KB)")






def load_df():
    creds = Credentials.from_service_account_file(
        str(BASE_DIR.parent / "config" / config.get("CREDENTIAL_PATH", "creds.json")),
        scopes=["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    gc = gspread.authorize(creds)
    ws = gc.open(config["SPREADSHEET_NAME"]).worksheet(config["SHEET_NAME"])
    data = ws.get_all_values()[1:]  # bỏ dòng header
    df = pd.DataFrame(data)
    df = df.rename(columns={0: "image_code", 20: "text_thumb", 21: "hook", 22: "highlights", 34: "channel"})
    df = df[df["channel"].str.contains("KA1-T4", na=False)]
    return df[["image_code", "text_thumb", "hook", "highlights"]]

def main():
    for _, row in load_df().iterrows():
        generate(row)

if __name__ == "__main__":
    main()
