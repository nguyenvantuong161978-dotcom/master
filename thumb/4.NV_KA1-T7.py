#!/usr/bin/env python3
"""
remove_bg_add_text.py – v4.3
────────────────────────────
• Chỉ skip khi <code>.png có metadata 'Processed=remove_bg_add_text_v4'
• Ảnh gốc đuôi .png (chưa metadata) vẫn được xử lý.
"""
from __future__ import annotations

import os, sys, json, math
from pathlib import Path
from typing import Optional

import gspread
from google.oauth2.service_account import Credentials
from rembg import remove
from PIL import (
    Image, ImageDraw, ImageFont, ImageOps, ImageFilter,
    PngImagePlugin
)
import numpy as np
import cv2
import torch
from torchvision import transforms

# ── ĐƯỜNG DẪN ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODNET_SRC = BASE_DIR / "MODNet" / "src"

if not MODNET_SRC.exists():
    print("❌ MODNet/src not found. Please check your folder structure.")
    sys.exit(1)

sys.path.append(str(MODNET_SRC))
from models.modnet import MODNet

# ── CONFIG ────────────────────────────────────────────────────────────
PIC_DIR      = BASE_DIR / "pic"
OUTPUT_DIR   = BASE_DIR / "nv"
FONT_PATH    = BASE_DIR.parent / "fonts" / "00034-UTM-AvoBold.ttf"
CONFIG_FILE  = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR.parent / "config" / "config.json")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

CREDENTIAL_PATH  = str(BASE_DIR.parent / "config" / config.get("CREDENTIAL_PATH", "creds.json"))
SPREADSHEET_NAME = config["SPREADSHEET_NAME"]
SHEET_NAME       = config["SHEET_NAME"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PIC_DIR.mkdir(exist_ok=True)

# CỘT (0-based): A=0, Z=25, AA=26, ... AI=34
COL_CODE  = 0       # Cột A: mã
COL_NAME  = 40      # Cột AO
COL_GROUP = 34      # Cột AI: đánh dấu nhóm

TARGET_GROUP = "KA1-T7"

# ── THÔNG SỐ HIỂN THỊ ─────────────────────────────────────────────────
FONT_SIZE         = 96
AVATAR_D          = 900
OVERLAP           = 60
PADDING_TOP       = 10
MAX_BADGE_W       = 600
BADGE_XPAD        = 80
BADGE_YPAD        = 24
BADGE_RADIUS      = 100
SHADOW_OFFSET     = (18, 18)
DOT_R             = 2
DOT_STEP          = 8

from PIL import ImageColor
def RGBA(hex, a=255): return ImageColor.getrgb(hex) + (a,)

# Bảng màu đồng bộ
TEXT_COLOR        = RGBA("#ffde59")       # chữ trắng
TEXT_SHADOW_COLOR = RGBA("#000000", 120)  # bóng chữ đen mờ
BADGE_COLOR       = RGBA("#9255f4")       # khung đen
BADGE_SHADOW_COLOR= RGBA("#ffde59")       # bóng vàng


TRANSPARENT       = (0, 0, 0, 0)
META_KEY          = "Processed"
META_VALUE        = "remove_bg_add_text_v4"
VALID_EXT         = {".png", ".jpg", ".jpeg", ".webp"}

# ── TẢI MODNET ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modnet = MODNet(backbone_pretrained=False)
state_dict = torch.load("MODNet/modnet_photographic_portrait_matting.pth", map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
modnet.load_state_dict(state_dict)
modnet.to(device).eval()

def preprocess_modnet(pil_image):
    im = pil_image.convert("RGB").resize((512, 512))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(im).unsqueeze(0)

# ── HÀM GOOGLE SHEET ──────────────────────────────────────────────────
def get_records():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(CREDENTIAL_PATH, scopes=scopes)
    ws = gspread.authorize(creds).open(SPREADSHEET_NAME).worksheet(SHEET_NAME)
    rows = ws.get_all_values()[1:]  # bỏ header

    out = []
    for r in rows:
        if len(r) <= max(COL_CODE, COL_GROUP):
            continue

        code = r[COL_CODE].strip()
        group_cell = r[COL_GROUP].strip().upper()  # cột AI
        if not code:
            continue

        # chỉ nhận khi AI == TARGET_GROUP
        if group_cell != TARGET_GROUP.upper():
            continue

        name = ""
        if len(r) > COL_NAME:
            name = r[COL_NAME].strip()
        if not name:
            name = code

        out.append((code, name))
    return out

# ── CÁC HÀM PHỤ ───────────────────────────────────────────────────────
def is_processed(p: Path) -> bool:
    if not p.exists() or p.suffix.lower() != ".png":
        return False
    try:
        with Image.open(p) as im:
            return im.info.get(META_KEY) == META_VALUE
    except:
        return False

def find_image(code: str) -> Optional[Path]:
    for p in PIC_DIR.glob(f"{code}.*"):
        if p.suffix.lower() in VALID_EXT and p.name != f"{code}.png":
            return p
    return None

def measure_text(label, font):
    tmp = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), label, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def fit_font_size(label, font_path, max_width, base_size):
    size = base_size
    while size > 10:
        font = ImageFont.truetype(font_path, size)
        text_w, _ = measure_text(label, font)
        if text_w <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(font_path, 10)

def rounded(d, rect, r, *, fill):
    if hasattr(d, "rounded_rectangle"):
        d.rounded_rectangle(rect, r, fill=fill)
    else:
        d.rectangle(rect, fill=fill)

def need_remove_alpha(im):
    return im.mode in ("RGB", "BGR") or (
        im.mode in ("RGBA", "LA") and max(im.split()[-1].getextrema()) > 250
    )

# ── TẠO CARD ──────────────────────────────────────────────────────────
def build_card(img_p: Path, label: str, font):
    with Image.open(img_p) as im:
        if need_remove_alpha(im):
            im = im.convert("RGB")
            im = im.resize((im.width * 6, im.height * 6), Image.LANCZOS)
            im_tensor = preprocess_modnet(im).to(device)
            with torch.no_grad():
                _, _, matte = modnet(im_tensor, True)
            matte_np = matte[0][0].cpu().numpy()
            matte_np = cv2.resize(matte_np, im.size, interpolation=cv2.INTER_AREA)
            fg = np.array(im).astype(np.uint8)
            alpha = (matte_np * 255).astype(np.uint8)
            im = Image.fromarray(np.dstack((fg, alpha)))
            im = im.filter(ImageFilter.SMOOTH_MORE)
            im = im.resize((im.width // 6, im.height // 6), Image.LANCZOS)
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            r, g, b, a = im.split()
            a = a.filter(ImageFilter.GaussianBlur(radius=3))
            im = Image.merge("RGBA", (r, g, b, a))

    # Crop về bounding box để loại bỏ phần transparent
    if im.mode == "RGBA":
        alpha = im.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            im = im.crop(bbox)

    # Scale để chiều cao = AVATAR_D
    scale = AVATAR_D / im.height
    new_w = int(im.width * scale)
    avatar = im.resize((new_w, AVATAR_D), Image.LANCZOS)
    ava_w, ava_h = avatar.size
    badge_w = ava_w
    text_max_width = badge_w - BADGE_XPAD * 2
    font = fit_font_size(label, str(FONT_PATH), text_max_width, FONT_SIZE)
    txt_w, txt_h = measure_text(label, font)
    badge_h = txt_h + BADGE_YPAD * 2

    cw = max(ava_w + PADDING_TOP * 2, badge_w + SHADOW_OFFSET[0]*2)
    ch = PADDING_TOP + ava_h + badge_h - OVERLAP + SHADOW_OFFSET[1] + PADDING_TOP
    cv = Image.new("RGBA", (cw, ch), TRANSPARENT)
    dr = ImageDraw.Draw(cv)

    avx = (cw - ava_w) // 2
    avy = PADDING_TOP
    cv.paste(avatar, (avx, avy), avatar)

    # vị trí ô bóng (dưới)
    pill_x0 = (cw - badge_w) // 2 + SHADOW_OFFSET[0]
    pill_y0 = avy + ava_h - OVERLAP + SHADOW_OFFSET[1]
    rounded(dr, (pill_x0, pill_y0, pill_x0 + badge_w, pill_y0 + badge_h),
            BADGE_RADIUS, fill=BADGE_SHADOW_COLOR)

    # vị trí ô chính (trên)
    mx0 = pill_x0 - SHADOW_OFFSET[0]   # <<< THÊM LẠI
    my0 = pill_y0 - SHADOW_OFFSET[1]   # <<< THÊM LẠI
    rounded(dr, (mx0, my0, mx0 + badge_w, my0 + badge_h),
            BADGE_RADIUS, fill=BADGE_COLOR)

    # chữ: tâm của badge chính
    cx, cy = mx0 + badge_w / 2, my0 + badge_h / 2  # <<< THÊM LẠI
    for dx, dy in [(2, 2), (1, 1)]:
        dr.text((cx + dx, cy + dy), label, font=font, fill=TEXT_SHADOW_COLOR, anchor="mm")
    dr.text((cx, cy), label, font=font, fill=TEXT_COLOR, anchor="mm")

    return cv

# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    if not FONT_PATH.exists():
        print(f"❌ Font not found: {FONT_PATH}")
        sys.exit(1)

    try:
        font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    except:
        print("❌ Error loading font.")
        sys.exit(1)

    for code, name in get_records():
        outp = OUTPUT_DIR / f"{code}.png"
        if is_processed(outp):
            print(f"[SKIP] {code}: ready")
            continue

        ip = find_image(code) or (PIC_DIR / f"{code}.png")
        if not ip.exists():
            print(f"[SKIP] {code}: no image")
            continue

        try:
            with Image.open(ip) as test_im:
                test_im.verify()
        except Exception:
            print(f"[ERR]  {code}: corrupted image")
            continue

        try:
            card = build_card(ip, name, font)
            meta = PngImagePlugin.PngInfo()
            meta.add_text(META_KEY, META_VALUE)
            try:
                card.save(outp, pnginfo=meta)
            except:
                card.convert("RGBA").save(outp, pnginfo=meta)
            print(f"[OK]   {outp.relative_to(BASE_DIR)}")
        except Exception as e:
            print(f"[ERR]  {code}: {e}")

if __name__ == "__main__":
    main()
