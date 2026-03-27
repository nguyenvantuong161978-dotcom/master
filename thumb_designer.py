#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE3 Thumb Designer - Visual thumbnail template designer
Run: python thumb_designer.py
Opens browser at http://localhost:5199
"""

import sys, os, io, json, base64, copy, threading, webbrowser
from pathlib import Path
from datetime import datetime

try:
    from flask import Flask, request, jsonify, render_template_string
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# ============ PATHS ============
TOOL_DIR   = Path(__file__).parent
THUMB_DIR  = TOOL_DIR / "thumb"
ASSETS_DIR = THUMB_DIR / "assets"
FONTS_DIR  = TOOL_DIR / "fonts"
UPLOAD_DIR = TOOL_DIR / "_designer_uploads"
CONFIG_FILE = TOOL_DIR / "config" / "config.json"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PORT = 5199

# ============ REMOVE BACKGROUND ============
_bg_remover = None   # lazy singleton
_rembg_cache = {}    # img_path -> PIL Image

def _load_bg_remover():
    """Load rembg or MODNet once, return callable(img) -> RGBA img."""
    global _bg_remover
    if _bg_remover is not None:
        return _bg_remover

    # Try rembg (pip install rembg)
    try:
        from rembg import remove as _rembg_fn
        import io as _io
        def _rembg_remover(img):
            buf = _io.BytesIO()
            img.save(buf, format='PNG')
            out = _rembg_fn(buf.getvalue())
            return Image.open(_io.BytesIO(out)).convert('RGBA')
        _bg_remover = _rembg_remover
        print("[remove_bg] Using rembg")
        return _bg_remover
    except ImportError:
        pass

    # Try MODNet (same as 4.NV_*.py scripts)
    try:
        import cv2 as _cv2, numpy as _np, torch as _torch
        from torchvision import transforms as _T
        _mdir = THUMB_DIR / "MODNet"
        _src  = _mdir / "src"
        _pth  = _mdir / "modnet_photographic_portrait_matting.pth"
        if not _src.exists() or not _pth.exists():
            raise FileNotFoundError("MODNet model not found")
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from models.modnet import MODNet as _MODNet
        _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        _mod = _MODNet(backbone_pretrained=False)
        _sd  = _torch.load(str(_pth), map_location=_dev)
        _sd  = {k.replace("module.", ""): v for k, v in _sd.items()}
        _mod.load_state_dict(_sd)
        _mod.to(_dev).eval()
        def _modnet_remover(img):
            im_rgb = img.convert("RGB")
            im_big = im_rgb.resize((im_rgb.width * 4, im_rgb.height * 4), Image.LANCZOS)
            tensor = _T.Compose([_T.ToTensor(), _T.Normalize((0.5,), (0.5,))])(
                im_big.resize((512, 512))).unsqueeze(0).to(_dev)
            with _torch.no_grad():
                _, _, matte = _mod(tensor, True)
            matte_np = _cv2.resize(matte[0][0].cpu().numpy(), im_big.size, interpolation=_cv2.INTER_AREA)
            fg    = _np.array(im_big).astype(_np.uint8)
            alpha = (matte_np * 255).astype(_np.uint8)
            result = Image.fromarray(_np.dstack((fg, alpha))).filter(ImageFilter.SMOOTH_MORE)
            result = result.resize(img.size, Image.LANCZOS).convert("RGBA")
            r, g, b, a = result.split()
            a = a.filter(ImageFilter.GaussianBlur(radius=2))
            return Image.merge("RGBA", (r, g, b, a))
        _bg_remover = _modnet_remover
        print("[remove_bg] Using MODNet")
        return _bg_remover
    except Exception as e:
        print(f"[remove_bg] Not available: {e}")
        _bg_remover = lambda img: img.convert('RGBA')
        return _bg_remover


def remove_bg_image(img, cache_key=None):
    """Remove background from PIL Image. Returns RGBA image."""
    if cache_key and cache_key in _rembg_cache:
        return _rembg_cache[cache_key]
    remover = _load_bg_remover()
    try:
        result = remover(img)
        if cache_key:
            _rembg_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  [remove_bg] Error: {e}")
        return img.convert('RGBA')

FONTS_MAP = {
    "UTM-AvoBold":      "00034-UTM-AvoBold.ttf",
    "Anton":            "Anton-Regular.ttf",
    "Bebas Neue":       "BebasNeue-Regular.ttf",
    "Inter Bold":       "Inter-Bold.ttf",
    "League Spartan":   "LeagueSpartan-ExtraBold.ttf",
    "Montserrat":       "Montserrat-VariableFont_wght.ttf",
    "Nunito":           "Nunito-VariableFont_wght.ttf",
    "Roboto Condensed": "RobotoCondensed-BlackItalic.ttf",
    "Zuume SemiBold":   "Zuume SemiBold.ttf",
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024


# ============ FLASK ROUTES ============

@app.route('/')
def index():
    fonts = list(FONTS_MAP.keys())
    return render_template_string(HTML_TEMPLATE, fonts=fonts)


@app.route('/upload-image', methods=['POST'])
def upload_image():
    f = request.files.get('file')
    save_as = request.form.get('save_as', 'preview')
    if not f:
        return jsonify({'error': 'No file'}), 400

    suffix = Path(f.filename).suffix.lower() or '.jpg'
    mime = 'jpeg' if suffix in ('.jpg', '.jpeg') else suffix[1:]

    if save_as.startswith('bg:'):
        template_name = save_as[3:].strip() or 'untitled'
        dest = ASSETS_DIR / f"THUMB_{template_name}_bg{suffix}"
        f.save(str(dest))
        with open(dest, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return jsonify({
            'url': f'data:image/{mime};base64,{b64}',
            'saved_path': str(dest),
            'asset_relative': f'assets/THUMB_{template_name}_bg{suffix}'
        })
    else:
        tmp = UPLOAD_DIR / f"photo_preview{suffix}"
        f.save(str(tmp))
        with open(tmp, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return jsonify({'url': f'data:image/{mime};base64,{b64}'})


@app.route('/preview', methods=['POST'])
def preview():
    try:
        design = request.json
        img = pil_render(design)
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=85)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return jsonify({'image': f'data:image/jpeg;base64,{b64}'})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/export', methods=['POST'])
def export():
    try:
        data = request.json
        design = data.get('design', {})
        template_name = data.get('template_name', 'KA1-T1').strip()
        py_code = generate_py(design, template_name)
        out_path = THUMB_DIR / f"THUMB_{template_name}.py"
        out_path.write_text(py_code, encoding='utf-8')
        return jsonify({'success': True, 'path': str(out_path)})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ============ PIL RENDERING ============

def get_pil_font(font_name, size):
    filename = FONTS_MAP.get(font_name)
    if filename:
        path = FONTS_DIR / filename
        if path.exists():
            try:
                return ImageFont.truetype(str(path), int(size))
            except Exception:
                pass
    return ImageFont.load_default()


def hex_rgba(h, a=255):
    h = str(h).lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    if len(h) != 6:
        return (128, 128, 128, a)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def wrap_pil(draw, text, font, max_w):
    words = text.split()
    lines, line = [], ''
    for w in words:
        t = f'{line} {w}'.strip()
        if draw.textlength(t, font=font) <= max_w:
            line = t
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines or [text]


def smart_wrap_2(draw, text, font, max_w):
    """Smart wrap into max 2 lines: prefer splitting at punctuation, else balanced."""
    if draw.textlength(text, font=font) <= max_w:
        return [text]
    # Try split at sentence punctuation (. ! ? ,) - prefer rightmost that fits
    best = None
    for sep in ['. ', '! ', '? ', ', ', '; ']:
        idx = -1
        while True:
            idx = text.find(sep, idx + 1)
            if idx < 0:
                break
            l1 = text[:idx + len(sep)].strip()
            l2 = text[idx + len(sep):].strip()
            if l1 and l2 and draw.textlength(l1, font=font) <= max_w and draw.textlength(l2, font=font) <= max_w:
                best = [l1, l2]
    if best:
        return best
    # Fallback: balanced split at word boundary closest to middle
    words = text.split()
    mid = len(text) // 2
    best_diff = len(text)
    best_split = None
    pos = 0
    for i, word in enumerate(words[:-1]):
        pos += len(word) + (1 if i > 0 else 0)
        diff = abs(pos - mid)
        if diff < best_diff:
            l1 = ' '.join(words[:i+1])
            l2 = ' '.join(words[i+1:])
            if draw.textlength(l1, font=font) <= max_w and draw.textlength(l2, font=font) <= max_w:
                best_diff = diff
                best_split = [l1, l2]
    if best_split:
        return best_split
    # Last fallback: basic word wrap
    return wrap_pil(draw, text, font, max_w)[:2]


def pil_render(design):
    cw = design.get('canvas', {}).get('w', 1920)
    ch = design.get('canvas', {}).get('h', 1080)
    canvas = Image.new('RGBA', (cw, ch), (20, 20, 30, 255))

    for el in design.get('elements', []):
        try:
            _pil_render_el(canvas, el, cw, ch)
        except Exception as e:
            print(f"  [render] {el.get('type')}: {e}")

    return canvas


def _pil_render_el(canvas, el, cw, ch):
    draw = ImageDraw.Draw(canvas)
    t = el['type']
    x = int(el.get('x', 0))
    y = int(el.get('y', 0))
    w = int(el.get('w', cw))
    h = int(el.get('h', ch))

    if t == 'background':
        img_path = el.get('image_path', '')
        if img_path:
            full = THUMB_DIR / img_path if not Path(img_path).is_absolute() else Path(img_path)
            if full.exists():
                bg = Image.open(full).convert('RGBA')
                bg = ImageOps.fit(bg, (cw, ch), Image.LANCZOS)
                canvas.paste(bg, (0, 0))
                return
        canvas.paste(Image.new('RGBA', (cw, ch), hex_rgba(el.get('color', '#1a1a2e'))), (0, 0))

    elif t == 'photo':
        # Try to load preview image (uploaded temp)
        img_path = el.get('preview_path', '')
        if img_path and Path(img_path).exists():
            img = Image.open(img_path).convert('RGBA')
            if el.get('remove_bg'):
                img = remove_bg_image(img, cache_key=img_path)
            scale = h / img.height
            nw = int(img.width * scale)
            img = img.resize((nw, h), Image.LANCZOS)
            if nw > w:
                cx = (nw - w) // 2
                img = img.crop((cx, 0, cx + w, h))
            canvas.paste(img, (x, y), img)
        else:
            draw.rectangle([x, y, x+w, y+h], fill=(80, 80, 120, 120))
            f = get_pil_font('UTM-AvoBold', 48)
            draw.text((x + w//2, y + h//2), 'PHOTO', fill=(200, 200, 200, 200), font=f, anchor='mm')

    elif t == 'text_main':
        text = el.get('sample_text', 'Text chính\ncủa thumbnail')
        transform = el.get('text_transform', 'none')
        if transform == 'upper': text = text.upper()
        elif transform == 'lower': text = text.lower()
        color = el.get('color', '#ffffff')
        hi_color = el.get('highlight_color', '#ffca22')
        font_name = el.get('font', 'UTM-AvoBold')
        min_sz = int(el.get('min_font', 36))
        max_sz = int(el.get('max_font', 200))
        spacing = int(el.get('line_spacing', 10))
        align = el.get('align', 'center')
        stroke_w    = int(el.get('stroke_width', 0))
        stroke_c    = el.get('stroke_color', '#000000')
        hi_stroke_w = int(el.get('hi_stroke_width', 0))
        hi_stroke_c = el.get('hi_stroke_color', '#000000')
        # Use sample_highlights for preview
        import re as _re
        raw_hi = el.get('sample_highlights', '')
        phrases = [p.strip() for p in raw_hi.split('|') if p.strip()]

        def _find_spans(full, phs):
            spans = []
            for ph in phs:
                ph = ph.strip()
                if not ph: continue
                toks = [_re.escape(t) for t in ph.split()]
                if not toks: continue
                f_, l_ = toks[0], toks[-1]
                if f_ and f_[0].isalnum(): f_ = r'\b' + f_
                if l_ and l_[-1].isalnum(): l_ = l_ + r'\b'
                mid = r'[^\w]*\s*'
                core = f_ if len(toks) == 1 else mid.join([f_, *toks[1:-1], l_])
                try: pat = _re.compile(core, _re.I | _re.S)
                except: pat = _re.compile(_re.escape(ph), _re.I | _re.S)
                for m in pat.finditer(full):
                    spans.append((m.start(), m.end()))
            spans.sort()
            merged = []
            for s, e in spans:
                if merged and s <= merged[-1][1]: merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
                else: merged.append((s, e))
            return merged

        def _draw_line_hi(draw, line, lx, ty, font, color, hi_color, spans, char_pos,
                          stroke_w=0, stroke_c='#000000', hi_stroke_w=0, hi_stroke_c='#000000'):
            segs, cur = [], 0
            for s, e in spans:
                s2, e2 = s - char_pos, e - char_pos
                ls, le = max(s2, 0), min(e2, len(line))
                if ls >= le: continue
                if ls > cur: segs.append((line[cur:ls], False))
                segs.append((line[ls:le], True))
                cur = le
            if cur < len(line): segs.append((line[cur:], False))
            if not segs: segs = [(line, False)]
            cx = lx
            for seg, hi in segs:
                c = hex_rgba(hi_color) if hi else hex_rgba(color)
                sw = hi_stroke_w if hi else stroke_w
                sc = hex_rgba(hi_stroke_c if hi else stroke_c)
                if sw > 0:
                    draw.text((cx, ty), seg, fill=c, font=font, stroke_width=sw, stroke_fill=sc)
                else:
                    draw.text((cx, ty), seg, fill=c, font=font)
                cx += int(draw.textlength(seg, font=font))

        # Draw semi-transparent box to show bounds
        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 10))
        canvas.alpha_composite(overlay, (x, y))

        for size in range(max_sz, min_sz - 1, -2):
            f = get_pil_font(font_name, size)
            lines = wrap_pil(draw, text, f, w - 20)
            total_h = len(lines) * size + max(0, len(lines) - 1) * spacing
            if total_h <= h - 20:
                ty = y + max(0, (h - total_h) // 2)
                full_txt = '\n'.join(lines)
                spans = _find_spans(full_txt, phrases)
                char_pos = 0
                for line in lines:
                    lw = draw.textlength(line, font=f)
                    if align == 'center': tx = x + max(0, (w - int(lw)) // 2)
                    elif align == 'right': tx = x + w - int(lw) - 10
                    else: tx = x + 10
                    _draw_line_hi(draw, line, tx, ty, f, color, hi_color, spans, char_pos,
                                  stroke_w, stroke_c, hi_stroke_w, hi_stroke_c)
                    ty += size + spacing
                    char_pos += len(line) + 1
                return
        f = get_pil_font(font_name, min_sz)
        lines = wrap_pil(draw, text, f, w - 20)
        ty = y + 10
        char_pos = 0
        full_txt = '\n'.join(lines)
        spans = _find_spans(full_txt, phrases)
        for line in lines:
            lw = draw.textlength(line, font=f)
            if align == 'center': tx = x + max(0, (w - int(lw)) // 2)
            elif align == 'right': tx = x + w - int(lw) - 10
            else: tx = x + 10
            _draw_line_hi(draw, line, tx, ty, f, color, hi_color, spans, char_pos,
                          stroke_w, stroke_c, hi_stroke_w, hi_stroke_c)
            ty += min_sz + spacing
            char_pos += len(line) + 1

    elif t == 'hook':
        text = el.get('sample_text', 'HOOK TEXT').upper()
        color = el.get('color', '#000000')
        bg_color = el.get('bg_color', '#ffca22')
        font_name = el.get('font', 'Anton')
        shape = el.get('shape', 'pill')
        stroke_w = int(el.get('stroke_width', 0))
        stroke_color = el.get('stroke_color', '#000000')
        fixed_size = int(el.get('font_size', 0))

        bg_rgba = hex_rgba(bg_color)
        if shape == 'pill' and hasattr(draw, 'rounded_rectangle'):
            r = min(h // 2, 30)
            draw.rounded_rectangle([x, y, x+w, y+h], radius=r, fill=bg_rgba)
        elif shape == 'none':
            pass
        else:
            draw.rectangle([x, y, x+w, y+h], fill=bg_rgba)

        cx = x + w // 2
        if fixed_size > 0:
            f = get_pil_font(font_name, fixed_size)
            tw = draw.textlength(text, font=f)
            if tw > w - 40:
                lines = smart_wrap_2(draw, text, f, w - 40)
                gap = max(4, fixed_size // 8)
                total_h = len(lines) * fixed_size + (len(lines) - 1) * gap
                start_cy = y + h // 2 - total_h // 2 + fixed_size // 2
                for i, line in enumerate(lines):
                    cy_line = start_cy + i * (fixed_size + gap)
                    if stroke_w > 0:
                        draw.text((cx, cy_line), line, fill=hex_rgba(stroke_color),
                                  font=f, anchor="mm", stroke_width=stroke_w,
                                  stroke_fill=hex_rgba(stroke_color))
                    draw.text((cx, cy_line), line, fill=hex_rgba(color), font=f, anchor="mm")
            else:
                cy = y + h // 2
                if stroke_w > 0:
                    draw.text((cx, cy), text, fill=hex_rgba(stroke_color),
                              font=f, anchor="mm", stroke_width=stroke_w,
                              stroke_fill=hex_rgba(stroke_color))
                draw.text((cx, cy), text, fill=hex_rgba(color), font=f, anchor="mm")
        else:
            # Auto-fit: find largest size that fits in max 2 lines
            for sz in range(h, 20, -2):
                f = get_pil_font(font_name, sz)
                tw = draw.textlength(text, font=f)
                if tw <= w - 40:
                    # Fits 1 line - check height
                    if sz > h - 10:
                        continue
                    cy = y + h // 2
                    if stroke_w > 0:
                        draw.text((cx, cy), text, fill=hex_rgba(stroke_color),
                                  font=f, anchor="mm", stroke_width=stroke_w,
                                  stroke_fill=hex_rgba(stroke_color))
                    draw.text((cx, cy), text, fill=hex_rgba(color), font=f, anchor="mm")
                    break
                else:
                    # Try 2 lines
                    lines = smart_wrap_2(draw, text, f, w - 40)
                    if len(lines) <= 2:
                        gap = max(4, sz // 8)
                        total_h = len(lines) * sz + (len(lines) - 1) * gap
                        if total_h <= h - 10:
                            start_cy = y + h // 2 - total_h // 2 + sz // 2
                            for i, line in enumerate(lines):
                                cy_line = start_cy + i * (sz + gap)
                                if stroke_w > 0:
                                    draw.text((cx, cy_line), line, fill=hex_rgba(stroke_color),
                                              font=f, anchor="mm", stroke_width=stroke_w,
                                              stroke_fill=hex_rgba(stroke_color))
                                draw.text((cx, cy_line), line, fill=hex_rgba(color), font=f, anchor="mm")
                            break

    elif t == 'text_box':
        text = el.get('sample_text', '')
        if not text:
            return
        color = el.get('color', '#ffffff')
        bg_color = el.get('bg_color', '#333333')
        font_name = el.get('font', 'UTM-AvoBold')
        size = int(el.get('font_size', 48))
        align = el.get('align', 'center')
        radius = int(el.get('border_radius', 10))

        bg_rgba = hex_rgba(bg_color)
        if radius > 0 and hasattr(draw, 'rounded_rectangle'):
            draw.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=bg_rgba)
        else:
            draw.rectangle([x, y, x+w, y+h], fill=bg_rgba)

        f = get_pil_font(font_name, size)
        lines = wrap_pil(draw, text, f, w - 20)
        gap = 6
        total_h = len(lines) * size + max(0, len(lines) - 1) * gap
        start_cy = y + h // 2 - total_h // 2 + size // 2
        for i, line in enumerate(lines):
            cy_line = start_cy + i * (size + gap)
            draw.text((x + w // 2, cy_line), line, fill=hex_rgba(color), font=f, anchor="mm")

    elif t == 'text_static':
        text = el.get('text', '')
        if not text:
            return
        color = el.get('color', '#ffffff')
        font_name = el.get('font', 'UTM-AvoBold')
        size = int(el.get('font_size', 60))
        align = el.get('align', 'center')
        f = get_pil_font(font_name, size)
        tw = draw.textlength(text, font=f)
        if align == 'center':
            tx = x + max(0, int((w - tw) // 2))
        elif align == 'right':
            tx = x + w - int(tw) - 10
        else:
            tx = x + 10
        ty = y + max(0, int((h - size) // 2))
        draw.text((tx, ty), text, fill=hex_rgba(color), font=f)


# ============ PY GENERATION ============

# Template for the generated THUMB_*.py file
# Uses <<<PLACEHOLDER>>> syntax to avoid f-string escaping issues
PY_TEMPLATE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# THUMB_<<<TEMPLATE_NAME>>>.py
# Generated by VE3 Thumb Designer - <<<DATE>>>
# Edit DESIGN config to customize layout and colors.
# Run: python THUMB_<<<TEMPLATE_NAME>>>.py <code>

import sys, io, re, unicodedata, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import gspread
from google.oauth2.service_account import Credentials

BASE_DIR    = Path(__file__).parent
VISUAL_DIR  = Path("D:/AUTO/VISUAL")
OUTPUT_DIR  = BASE_DIR / "thumbnails"
FONTS_DIR   = BASE_DIR.parent / "fonts"
CONFIG_FILE = BASE_DIR.parent / "config" / "config.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# DESIGN CONFIG (generated by Thumb Designer - edit to customize)
# ===========================================================================
DESIGN = <<<DESIGN_JSON>>>
# ===========================================================================

FONTS_MAP = {
    "UTM-AvoBold":      FONTS_DIR / "00034-UTM-AvoBold.ttf",
    "Anton":            FONTS_DIR / "Anton-Regular.ttf",
    "Bebas Neue":       FONTS_DIR / "BebasNeue-Regular.ttf",
    "Inter Bold":       FONTS_DIR / "Inter-Bold.ttf",
    "League Spartan":   FONTS_DIR / "LeagueSpartan-ExtraBold.ttf",
    "Montserrat":       FONTS_DIR / "Montserrat-VariableFont_wght.ttf",
    "Nunito":           FONTS_DIR / "Nunito-VariableFont_wght.ttf",
    "Roboto Condensed": FONTS_DIR / "RobotoCondensed-BlackItalic.ttf",
    "Zuume SemiBold":   FONTS_DIR / "Zuume SemiBold.ttf",
}


# ============ GOOGLE SHEETS ============
def load_record(code, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cred_path = CONFIG_FILE.parent / cfg.get("CREDENTIAL_PATH", "creds.json")
            creds = Credentials.from_service_account_file(str(cred_path), scopes=[
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
            ])
            gc = gspread.authorize(creds)
            ws = gc.open("<<<SHEET_NAME>>>").worksheet("<<<INPUT_SHEET>>>")
            rows = ws.get_all_values()[1:]
            for r in rows:
                if r and r[0].strip() == code:
                    return {
                        "text_thumb": r[20].strip() if len(r) > 20 else "",
                        "hook":       r[21].strip() if len(r) > 21 else "",
                        "highlights": r[22].strip() if len(r) > 22 else "",
                    }
            return {"text_thumb": "", "hook": "", "highlights": ""}
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"[WARN] Sheets error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  Retry in {wait}s...")
                time.sleep(wait)
    print(f"[ERROR] Sheets failed after {max_retries} attempts. Cannot proceed without data.")
    return None


# ============ HELPERS ============
def get_font(name, size):
    path = FONTS_MAP.get(name)
    if path and Path(path).exists():
        try:
            return ImageFont.truetype(str(path), int(size))
        except Exception:
            pass
    return ImageFont.load_default()


def hex_rgba(h, a=255):
    h = str(h).lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (128, 128, 128, a)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    return (text.strip()
            .replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "-"))


def wrap_text(draw, text, font, max_w):
    words = text.split()
    lines, line = [], ""
    for w in words:
        t = f"{line} {w}".strip()
        if draw.textlength(t, font=font) <= max_w:
            line = t
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines or [text]


def smart_wrap_2(draw, text, font, max_w):
    if draw.textlength(text, font=font) <= max_w:
        return [text]
    best = None
    for sep in [". ", "! ", "? ", ", ", "; "]:
        idx = -1
        while True:
            idx = text.find(sep, idx + 1)
            if idx < 0:
                break
            l1 = text[:idx + len(sep)].strip()
            l2 = text[idx + len(sep):].strip()
            if l1 and l2 and draw.textlength(l1, font=font) <= max_w and draw.textlength(l2, font=font) <= max_w:
                best = [l1, l2]
    if best:
        return best
    words = text.split()
    mid = len(text) // 2
    best_diff = len(text)
    best_split = None
    pos = 0
    for i, word in enumerate(words[:-1]):
        pos += len(word) + (1 if i > 0 else 0)
        diff = abs(pos - mid)
        if diff < best_diff:
            l1 = " ".join(words[:i+1])
            l2 = " ".join(words[i+1:])
            if draw.textlength(l1, font=font) <= max_w and draw.textlength(l2, font=font) <= max_w:
                best_diff = diff
                best_split = [l1, l2]
    if best_split:
        return best_split
    return wrap_text(draw, text, font, max_w)[:2]


def split_phrases(raw):
    return [p.strip() for p in str(raw).split("|") if p.strip()]


def find_highlight_spans(full_text, phrases):
    spans = []
    for ph in phrases:
        ph = ph.strip()
        if not ph:
            continue
        tokens = [re.escape(t) for t in ph.split()]
        if not tokens:
            continue
        first, last = tokens[0], tokens[-1]
        if first and first[0].isalnum():
            first = r"\b" + first
        if last and last[-1].isalnum():
            last = last + r"\b"
        mid = r"[^\w]*\s*"
        core = first if len(tokens) == 1 else mid.join([first, *tokens[1:-1], last])
        try:
            pattern = re.compile(core, re.I | re.S)
        except re.error:
            pattern = re.compile(re.escape(ph), re.I | re.S)
        for m in pattern.finditer(full_text):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged


def draw_rich_text(draw, lines, x, y, w, font, color, hi_color, spacing, align, spans,
                   stroke_w=0, stroke_c="#000000", hi_stroke_w=0, hi_stroke_c="#000000"):
    ty = y
    char_pos = 0
    for line in lines:
        lw = int(draw.textlength(line, font=font))
        if align == "center":
            lx = x + max(0, (w - lw) // 2)
        elif align == "right":
            lx = x + w - lw - 10
        else:
            lx = x + 10

        segs, cur = [], 0
        for s, e in spans:
            s2 = s - char_pos
            e2 = e - char_pos
            ls = max(s2, 0)
            le = min(e2, len(line))
            if ls >= le:
                continue
            if ls > cur:
                segs.append((line[cur:ls], False))
            segs.append((line[ls:le], True))
            cur = le
        if cur < len(line):
            segs.append((line[cur:], False))
        if not segs:
            segs = [(line, False)]

        cx = lx
        for seg, hi in segs:
            c = hex_rgba(hi_color) if hi else hex_rgba(color)
            sw = hi_stroke_w if hi else stroke_w
            sc = hex_rgba(hi_stroke_c if hi else stroke_c)
            if sw > 0:
                draw.text((cx, ty), seg, fill=c, font=font, stroke_width=sw, stroke_fill=sc)
            else:
                draw.text((cx, ty), seg, fill=c, font=font)
            cx += int(draw.textlength(seg, font=font))

        ty += font.size + spacing
        char_pos += len(line) + 1


def save_jpeg(img, path, max_bytes=2_000_000):
    rgb = img.convert("RGB")
    for _ in range(4):
        for q in range(95, 49, -5):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=q, optimize=True,
                     progressive=True, subsampling="4:2:0")
            if buf.tell() <= max_bytes:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(buf.getvalue())
                return
        new_w = int(rgb.width * 0.9)
        new_h = int(rgb.height * 0.9)
        if new_w < 640:
            break
        rgb = rgb.resize((new_w, new_h), Image.LANCZOS)
    rgb.save(path, format="JPEG", quality=50, optimize=True, progressive=True)


# ============ BACKGROUND REMOVAL ============
def remove_bg_image(img):
    """Remove background using rembg or MODNet. Returns RGBA image."""
    import io as _io
    # Try rembg (pip install rembg)
    try:
        from rembg import remove as _rembg
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        out = _rembg(buf.getvalue())
        return Image.open(_io.BytesIO(out)).convert("RGBA")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [remove_bg] rembg: {e}")
    # Try MODNet (same as 4.NV_*.py)
    try:
        import sys as _sys, cv2 as _cv2, numpy as _np, torch as _torch
        from torchvision import transforms as _T
        _mdir = BASE_DIR / "MODNet"
        _src  = _mdir / "src"
        _pth  = _mdir / "modnet_photographic_portrait_matting.pth"
        if not _src.exists() or not _pth.exists():
            raise FileNotFoundError("MODNet not found")
        if str(_src) not in _sys.path:
            _sys.path.insert(0, str(_src))
        from models.modnet import MODNet as _MODNet
        _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        _m = _MODNet(backbone_pretrained=False)
        _sd = _torch.load(str(_pth), map_location=_dev)
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        _m.load_state_dict(_sd); _m.to(_dev).eval()
        im_rgb = img.convert("RGB")
        im_big = im_rgb.resize((im_rgb.width * 4, im_rgb.height * 4), Image.LANCZOS)
        tensor = _T.Compose([_T.ToTensor(), _T.Normalize((0.5,), (0.5,))])(
            im_big.resize((512, 512))).unsqueeze(0).to(_dev)
        with _torch.no_grad():
            _, _, matte = _m(tensor, True)
        matte_np = _cv2.resize(matte[0][0].cpu().numpy(), im_big.size, interpolation=_cv2.INTER_AREA)
        fg    = _np.array(im_big).astype(_np.uint8)
        alpha = (matte_np * 255).astype(_np.uint8)
        result = Image.fromarray(_np.dstack((fg, alpha))).filter(ImageFilter.SMOOTH_MORE)
        result = result.resize(img.size, Image.LANCZOS).convert("RGBA")
        r, g, b, a = result.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=2))
        return Image.merge("RGBA", (r, g, b, a))
    except Exception as e:
        print(f"  [remove_bg] MODNet: {e}")
    return img.convert("RGBA")


# ============ RENDER ELEMENTS ============
def render_element(canvas, el, code, data):
    draw = ImageDraw.Draw(canvas)
    cw, ch = canvas.size
    t = el["type"]
    x = int(el.get("x", 0))
    y = int(el.get("y", 0))
    w = int(el.get("w", cw))
    h = int(el.get("h", ch))

    if t == "background":
        img_path = el.get("image_path", "")
        if img_path:
            full = BASE_DIR / img_path if not Path(img_path).is_absolute() else Path(img_path)
            if full.exists():
                bg = Image.open(full).convert("RGBA")
                bg = ImageOps.fit(bg, (cw, ch), Image.LANCZOS)
                canvas.paste(bg, (0, 0))
                return
        canvas.paste(Image.new("RGBA", (cw, ch), hex_rgba(el.get("color", "#1a1a2e"))), (0, 0))

    elif t == "photo":
        thumb_dir = VISUAL_DIR / code / "thumbnail"
        nv_path = VISUAL_DIR / code / "nv" / "nv1.png"
        thumb_candidates = [thumb_dir / f"thumb_{i:03d}.png" for i in <<<THUMB_ORDER>>>]

        photo_path = None
        if el.get("remove_bg"):
            # Prefer nv1.png, fallback to thumb images
            candidates = [nv_path] + thumb_candidates
            for c in candidates:
                if c.exists():
                    photo_path = c; break
        else:
            # Prefer thumb images, fallback to nv1.png
            candidates = thumb_candidates + [nv_path]
            for c in candidates:
                if c.exists():
                    photo_path = c; break

        if photo_path is None:
            print(f"  [WARN] No photo found for {code} (checked thumbnail/ and nv/)")
            return

        print(f"  [photo] Using: {photo_path.name}")
        img = Image.open(photo_path).convert("RGBA")
        if el.get("remove_bg") and photo_path == nv_path:
            img = remove_bg_image(img)
        scale = h / img.height
        nw = int(img.width * scale)
        img = img.resize((nw, h), Image.LANCZOS)
        if nw > w:
            cx = (nw - w) // 2
            img = img.crop((cx, 0, cx + w, h))
        canvas.paste(img, (x, y), img)

    elif t == "text_main":
        text = clean_text(data.get("text_thumb", ""))
        if not text:
            return
        transform = el.get("text_transform", "none")
        if transform == "upper": text = text.upper()
        elif transform == "lower": text = text.lower()
        color       = el.get("color", "#ffffff")
        hi_color    = el.get("highlight_color", "#ffca22")
        font_name   = el.get("font", "UTM-AvoBold")
        min_sz      = int(el.get("min_font", 36))
        max_sz      = int(el.get("max_font", 200))
        spacing     = int(el.get("line_spacing", 10))
        align       = el.get("align", "center")
        stroke_w    = int(el.get("stroke_width", 0))
        stroke_c    = el.get("stroke_color", "#000000")
        hi_stroke_w = int(el.get("hi_stroke_width", 0))
        hi_stroke_c = el.get("hi_stroke_color", "#000000")
        phrases     = split_phrases(data.get("highlights", ""))

        for size in range(max_sz, min_sz - 1, -2):
            f = get_font(font_name, size)
            lines = wrap_text(draw, text, f, w - 20)
            total_h = len(lines) * size + max(0, len(lines) - 1) * spacing
            if total_h <= h - 20:
                ty = y + max(0, (h - total_h) // 2)
                full_txt = "\n".join(lines)
                spans = find_highlight_spans(full_txt, phrases)
                draw_rich_text(draw, lines, x, ty, w, f, color, hi_color, spacing, align, spans,
                               stroke_w, stroke_c, hi_stroke_w, hi_stroke_c)
                return
        f = get_font(font_name, min_sz)
        lines = wrap_text(draw, text, f, w - 20)
        full_txt = "\n".join(lines)
        spans = find_highlight_spans(full_txt, phrases)
        draw_rich_text(draw, lines, x, y + 10, w, f, color, hi_color, spacing, align, spans,
                       stroke_w, stroke_c, hi_stroke_w, hi_stroke_c)

    elif t == "hook":
        text = clean_text(data.get("hook", "")).upper()
        if not text:
            return
        color        = el.get("color", "#000000")
        bg_color     = el.get("bg_color", "#ffca22")
        font_name    = el.get("font", "Anton")
        shape        = el.get("shape", "pill")
        stroke_w     = int(el.get("stroke_width", 0))
        stroke_color = el.get("stroke_color", "#000000")
        fixed_size   = int(el.get("font_size", 0))

        bg_rgba = hex_rgba(bg_color)
        if shape == "pill" and hasattr(draw, "rounded_rectangle"):
            r = min(h // 2, 40)
            draw.rounded_rectangle([x, y, x+w, y+h], radius=r, fill=bg_rgba)
        elif shape == "rect":
            draw.rectangle([x, y, x+w, y+h], fill=bg_rgba)

        cx = x + w // 2
        if fixed_size > 0:
            f = get_font(font_name, fixed_size)
            tw = int(draw.textlength(text, font=f))
            if tw > w - 40:
                lines = smart_wrap_2(draw, text, f, w - 40)
                gap = max(4, fixed_size // 8)
                total_h = len(lines) * fixed_size + (len(lines) - 1) * gap
                start_cy = y + h // 2 - total_h // 2 + fixed_size // 2
                for i, line in enumerate(lines):
                    cy_line = start_cy + i * (fixed_size + gap)
                    if stroke_w > 0:
                        draw.text((cx, cy_line), line, fill=hex_rgba(stroke_color),
                                  font=f, anchor="mm", stroke_width=stroke_w,
                                  stroke_fill=hex_rgba(stroke_color))
                    draw.text((cx, cy_line), line, fill=hex_rgba(color), font=f, anchor="mm")
            else:
                cy = y + h // 2
                if stroke_w > 0:
                    draw.text((cx, cy), text, fill=hex_rgba(stroke_color),
                              font=f, anchor="mm", stroke_width=stroke_w,
                              stroke_fill=hex_rgba(stroke_color))
                draw.text((cx, cy), text, fill=hex_rgba(color), font=f, anchor="mm")
        else:
            # Auto-fit: largest size that fits max 2 lines
            for sz in range(h, 20, -2):
                f = get_font(font_name, sz)
                tw = int(draw.textlength(text, font=f))
                if tw <= w - 40:
                    if sz > h - 10:
                        continue
                    cy = y + h // 2
                    if stroke_w > 0:
                        draw.text((cx, cy), text, fill=hex_rgba(stroke_color),
                                  font=f, anchor="mm", stroke_width=stroke_w,
                                  stroke_fill=hex_rgba(stroke_color))
                    draw.text((cx, cy), text, fill=hex_rgba(color), font=f, anchor="mm")
                    break
                else:
                    lines = smart_wrap_2(draw, text, f, w - 40)
                    if len(lines) <= 2:
                        gap = max(4, sz // 8)
                        total_h = len(lines) * sz + (len(lines) - 1) * gap
                        if total_h <= h - 10:
                            start_cy = y + h // 2 - total_h // 2 + sz // 2
                            for i, line in enumerate(lines):
                                cy_line = start_cy + i * (sz + gap)
                                if stroke_w > 0:
                                    draw.text((cx, cy_line), line, fill=hex_rgba(stroke_color),
                                              font=f, anchor="mm", stroke_width=stroke_w,
                                              stroke_fill=hex_rgba(stroke_color))
                                draw.text((cx, cy_line), line, fill=hex_rgba(color), font=f, anchor="mm")
                            break

    elif t == "text_box":
        text = el.get("sample_text", "")
        if not text:
            return
        color      = el.get("color", "#ffffff")
        bg_color   = el.get("bg_color", "#333333")
        font_name  = el.get("font", "UTM-AvoBold")
        size       = int(el.get("font_size", 48))
        align      = el.get("align", "center")
        radius     = int(el.get("border_radius", 10))

        bg_rgba = hex_rgba(bg_color)
        if radius > 0 and hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=bg_rgba)
        else:
            draw.rectangle([x, y, x+w, y+h], fill=bg_rgba)

        f = get_font(font_name, size)
        lines = wrap_text(draw, text, f, w - 20)
        gap = 6
        total_h = len(lines) * size + max(0, len(lines) - 1) * gap
        start_cy = y + h // 2 - total_h // 2 + size // 2
        for i, line in enumerate(lines):
            cy_line = start_cy + i * (size + gap)
            draw.text((x + w // 2, cy_line), line, fill=hex_rgba(color), font=f, anchor="mm")

    elif t == "text_static":
        text = el.get("text", "")
        if not text:
            return
        color     = el.get("color", "#ffffff")
        font_name = el.get("font", "UTM-AvoBold")
        size      = int(el.get("font_size", 60))
        align     = el.get("align", "center")
        f = get_font(font_name, size)
        tw = int(draw.textlength(text, font=f))
        if align == "center":
            tx = x + max(0, (w - tw) // 2)
        elif align == "right":
            tx = x + w - tw - 10
        else:
            tx = x + 10
        ty3 = y + max(0, (h - size) // 2)
        draw.text((tx, ty3), text, fill=hex_rgba(color), font=f)


# ============ MAIN ============
def main():
    code = sys.argv[1] if len(sys.argv) > 1 else None
    if not code:
        print("Usage: python THUMB_<<<TEMPLATE_NAME>>>.py <code>")
        sys.exit(1)

    out_path = OUTPUT_DIR / f"{code}.jpg"
    if out_path.exists():
        print(f"  Skip {code} (already exists)")
        return

    print(f"  Processing: {code}")
    data = load_record(code)
    if data is None:
        print(f"  SKIP {code}: không lấy được dữ liệu trang tính sau nhiều lần thử")
        return

    cw = DESIGN["canvas"]["w"]
    ch = DESIGN["canvas"]["h"]
    canvas = Image.new("RGBA", (cw, ch), (20, 20, 30, 255))

    for el in DESIGN.get("elements", []):
        try:
            render_element(canvas, el, code, data)
        except Exception as e:
            print(f"  [WARN] Element {el.get('type')}: {e}")

    save_jpeg(canvas, out_path)
    size_kb = out_path.stat().st_size // 1024 if out_path.exists() else 0
    print(f"  Saved: {out_path.name} ({size_kb} KB)")


if __name__ == "__main__":
    main()
'''


def generate_py(design, template_name):
    export_design = copy.deepcopy(design)
    # Remove browser-only fields
    for el in export_design.get('elements', []):
        el.pop('src', None)
        el.pop('preview_path', None)

    config_json = json.dumps(export_design, indent=4, ensure_ascii=False)
    # JSON booleans → Python booleans
    config_json = config_json.replace(': true', ': True').replace(': false', ': False')

    config_data = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

    sheet_name  = config_data.get('SPREADSHEET_NAME', 'KA')
    input_sheet = config_data.get('SHEET_NAME', 'INPUT')

    thumb_order = design.get('thumb_order', '[4,5,6,1,2,3]')

    py = PY_TEMPLATE
    py = py.replace('<<<TEMPLATE_NAME>>>', template_name)
    py = py.replace('<<<DATE>>>', datetime.now().strftime("%Y-%m-%d %H:%M"))
    py = py.replace('<<<DESIGN_JSON>>>', config_json)
    py = py.replace('<<<SHEET_NAME>>>', sheet_name)
    py = py.replace('<<<INPUT_SHEET>>>', input_sheet)
    py = py.replace('<<<THUMB_ORDER>>>', thumb_order)
    return py


# ============ HTML TEMPLATE ============
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>VE3 Thumb Designer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: #12121f; color: #dde; display: flex; flex-direction: column; height: 100vh; overflow: hidden; font-size: 13px; }
#toolbar { background: #1a1a30; border-bottom: 1px solid #2a2a50; padding: 8px 16px; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
#toolbar span.logo { font-weight: 700; font-size: 16px; color: #6aacff; margin-right: 8px; }
#toolbar label { font-size: 12px; color: #888; }
#toolbar input[type=text] { background: #0f0f25; border: 1px solid #3a3a70; border-radius: 5px; color: #fff; padding: 5px 10px; width: 150px; }
.btn { background: #3a6fd8; color: #fff; border: none; border-radius: 5px; padding: 6px 14px; cursor: pointer; font-size: 12px; font-weight: 600; transition: background .15s; white-space: nowrap; }
.btn:hover { background: #2a5fc8; }
.btn.green { background: #27a060; } .btn.green:hover { background: #1e8050; }
.btn.purple { background: #7040c0; } .btn.purple:hover { background: #5030a0; }
#main { display: flex; flex: 1; overflow: hidden; }

/* LEFT */
#left-panel { width: 180px; background: #1a1a30; border-right: 1px solid #2a2a50; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
#left-panel .section-title { padding: 8px 12px; font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #2a2a50; }
.add-btn { display: block; margin: 4px 8px; padding: 7px 10px; background: #0f0f25; border: 1px dashed #3a3a70; border-radius: 5px; cursor: pointer; font-size: 12px; color: #99a; transition: all .15s; text-align: left; width: calc(100% - 16px); }
.add-btn:hover { background: #1a1a50; border-color: #6aacff; color: #fff; }
#element-list { flex: 1; overflow-y: auto; }
.el-item { padding: 7px 12px; cursor: pointer; font-size: 12px; border-left: 3px solid transparent; display: flex; align-items: center; gap: 4px; user-select: none; }
.el-item:hover { background: #1a1a40; }
.el-item.selected { background: #1a2a50; border-left-color: #6aacff; color: #fff; }
.el-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.el-btn { color: #555; cursor: pointer; padding: 1px 3px; border-radius: 3px; font-size: 11px; display: none; }
.el-btn:hover { color: #fff; background: #3a3a60; }
.el-item:hover .el-btn { display: inline; }

/* CANVAS */
#canvas-area { flex: 1; display: flex; align-items: center; justify-content: center; background: #0a0a18; overflow: auto; }
#design-canvas { display: block; cursor: default; outline: none; box-shadow: 0 0 40px rgba(0,0,0,.8); }

/* RIGHT */
#right-panel { width: 250px; background: #1a1a30; border-left: 1px solid #2a2a50; overflow-y: auto; flex-shrink: 0; }
#right-panel .section-title { padding: 8px 12px; font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #2a2a50; position: sticky; top: 0; background: #1a1a30; z-index: 1; }
.pg { padding: 10px 12px; border-bottom: 1px solid #1e1e3a; }
.pg label { display: block; font-size: 10px; color: #777; margin-bottom: 4px; text-transform: uppercase; letter-spacing: .5px; }
.pg input[type=text], .pg input[type=number], .pg textarea, .pg select { width: 100%; background: #0f0f25; border: 1px solid #2a2a50; border-radius: 4px; color: #dde; padding: 5px 7px; font-size: 12px; }
.pg textarea { resize: vertical; min-height: 54px; }
.pg input[type=color] { width: 100%; height: 30px; border: 1px solid #2a2a50; border-radius: 4px; background: #0f0f25; cursor: pointer; padding: 2px; }
.row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 6px; }
.row4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 4px; }
.row4 input { background: #0f0f25; border: 1px solid #2a2a50; border-radius: 3px; color: #dde; padding: 4px 2px; text-align: center; font-size: 11px; }
.row4 .lbl { font-size: 9px; color: #666; text-align: center; margin-top: 1px; }
#no-sel { padding: 20px 12px; color: #555; font-size: 12px; }

/* STATUS */
#statusbar { background: #0a0a18; border-top: 1px solid #2a2a50; padding: 3px 14px; font-size: 10px; color: #555; flex-shrink: 0; }

/* OVERLAY */
#overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.9); z-index: 200; align-items: center; justify-content: center; flex-direction: column; gap: 12px; }
#overlay.show { display: flex; }
#prev-img { max-width: 90vw; max-height: 82vh; border-radius: 6px; }
#msg-banner { display: none; position: fixed; top: 60px; right: 20px; background: #27a060; color: #fff; padding: 8px 16px; border-radius: 6px; font-size: 13px; z-index: 300; }

input[type=file] { display: none; }
</style>
</head>
<body>

<div id="toolbar">
  <span class="logo">🎨 Thumb Designer</span>
  <label>Template:</label>
  <input type="text" id="tname" value="KA1-T1" placeholder="KA1-T5, TA2-T1, DT1-T1...">
  <label>Thumb:</label>
  <select id="thumb_order" style="padding:4px;border-radius:4px;background:#2a2a3e;color:#fff;border:1px solid #444">
    <option value="[4,5,6,1,2,3]">004 truoc (KA/TA)</option>
    <option value="[1,2,3,4,5,6]">001 truoc (chu de moi)</option>
  </select>
  <button class="btn" onclick="doPreview()">👁 Preview PIL</button>
  <button class="btn green" onclick="doExport()">💾 Export .py</button>
  <button class="btn purple" onclick="saveJSON()">📥 Save JSON</button>
  <button class="btn" style="background:#444" onclick="loadJSON()">📂 Load JSON</button>
  <button class="btn" style="background:#3a5540" onclick="loadPY()">📄 Load .py</button>
</div>

<div id="main">
  <div id="left-panel">
    <div class="section-title">Thêm phần tử</div>
    <button class="add-btn" onclick="addEl('background')">🖼 Background</button>
    <button class="add-btn" onclick="addEl('photo')">👤 Ảnh minh họa</button>
    <button class="add-btn" onclick="addEl('text_main')">📝 Text chính</button>
    <button class="add-btn" onclick="addEl('hook')">🔥 Hook bar</button>
    <button class="add-btn" onclick="addEl('text_box')">⬜ Khung chữ</button>
    <button class="add-btn" onclick="addEl('text_static')">💬 Text tĩnh</button>
    <div class="section-title" style="margin-top:6px;">Các lớp (trên → dưới)</div>
    <div id="element-list"></div>
  </div>

  <div id="canvas-area">
    <canvas id="design-canvas" tabindex="0"></canvas>
  </div>

  <div id="right-panel">
    <div class="section-title">Thuộc tính</div>
    <div id="props"></div>
  </div>
</div>

<div id="statusbar">Sẵn sàng</div>
<div id="overlay" onclick="this.classList.remove('show')">
  <img id="prev-img" src="" alt="Preview">
  <span style="color:#666;font-size:11px;">Click để đóng</span>
</div>
<div id="msg-banner"></div>
<input type="file" id="fi-bg" accept="image/*">
<input type="file" id="fi-photo" accept="image/*">

<script>
// ========== STATE ==========
const CW = 1920, CH = 1080;
let SCALE = 0.5;
let canvas, ctx;
let design = { canvas: {w:1920,h:1080}, elements: [] };
let selId = null;
let drag = null;
let uid = 0;
const FONTS = {{ fonts | tojson }};
const IMG_CACHE = {};  // id -> {src, img}

// ========== INIT ==========
window.addEventListener('load', () => {
  canvas = document.getElementById('design-canvas');
  ctx = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  canvas.addEventListener('mousedown', onDown);
  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('mouseup', onUp);
  canvas.addEventListener('dblclick', onDbl);
  canvas.addEventListener('keydown', onKey);
  document.getElementById('fi-bg').addEventListener('change', onBgUpload);
  document.getElementById('fi-photo').addEventListener('change', onPhotoUpload);
  draw();
  renderList();
  renderProps();
  status('Sẵn sàng. Thêm phần tử từ panel trái.');
});

function resizeCanvas() {
  const area = document.getElementById('canvas-area');
  const maxW = area.clientWidth - 40;
  const maxH = area.clientHeight - 40;
  SCALE = Math.min(maxW / CW, maxH / CH, 0.62);
  canvas.width  = Math.round(CW * SCALE);
  canvas.height = Math.round(CH * SCALE);
  draw();
}

// ========== ELEMENT DEFAULTS ==========
const EL_ICON = {background:'🖼',photo:'👤',text_main:'📝',hook:'🔥',text_box:'⬜',text_static:'💬'};
const EL_LABEL = {background:'Background',photo:'Ảnh minh họa',text_main:'Text chính',hook:'Hook bar',text_box:'Khung chữ',text_static:'Text tĩnh'};

function mkEl(type) {
  const id = ++uid;
  switch(type) {
    case 'background':  return {id,type,x:0,y:0,w:1920,h:1080,color:'#1a1a2e',image_path:'',src:null};
    case 'photo':       return {id,type,x:1200,y:0,w:720,h:1080,src:null,preview_path:'',remove_bg:false};
    case 'text_main':   return {id,type,x:20,y:20,w:1150,h:860,
                                sample_text:'Text chính của thumbnail',
                                sample_highlights:'',
                                font:'UTM-AvoBold',color:'#ffffff',highlight_color:'#ffca22',
                                min_font:36,max_font:200,line_spacing:10,align:'center',
                                text_transform:'none',
                                stroke_width:0,stroke_color:'#000000',
                                hi_stroke_width:0,hi_stroke_color:'#000000'};
    case 'hook':        return {id,type,x:20,y:880,w:1150,h:180,
                                sample_text:'HOOK TEXT',font:'Anton',
                                font_size:0,color:'#000000',bg_color:'#ffca22',shape:'pill',
                                stroke_width:0,stroke_color:'#000000'};
    case 'text_box':    return {id,type,x:100,y:100,w:500,h:120,
                                sample_text:'Text here',font:'UTM-AvoBold',
                                font_size:48,color:'#ffffff',bg_color:'#333333',
                                border_radius:10,align:'center'};
    case 'text_static': return {id,type,x:20,y:20,w:600,h:80,
                                text:'Static text',font:'UTM-AvoBold',
                                font_size:60,color:'#ffffff',align:'center'};
  }
  return {id,type,x:100,y:100,w:400,h:200};
}

// ========== CRUD ==========
function addEl(type) {
  const el = mkEl(type);
  design.elements.push(el);
  selId = el.id;
  draw(); renderList(); renderProps();
}
function delEl(id) {
  design.elements = design.elements.filter(e => e.id !== id);
  if (selId === id) { selId = null; renderProps(); }
  draw(); renderList();
}
function moveUp(id) {
  const i = design.elements.findIndex(e => e.id === id);
  if (i < design.elements.length-1) {
    [design.elements[i], design.elements[i+1]] = [design.elements[i+1], design.elements[i]];
    draw(); renderList();
  }
}
function moveDn(id) {
  const i = design.elements.findIndex(e => e.id === id);
  if (i > 0) {
    [design.elements[i], design.elements[i-1]] = [design.elements[i-1], design.elements[i]];
    draw(); renderList();
  }
}
function selEl(id) { selId = id; draw(); renderList(); renderProps(); }

// ========== DRAWING ==========
function getImg(el) {
  if (!el.src) return null;
  const c = IMG_CACHE[el.id];
  if (c && c.src === el.src) return c.img.complete ? c.img : null;
  const img = new Image();
  img.onload = () => draw();
  img.src = el.src;
  IMG_CACHE[el.id] = {src: el.src, img};
  return null;
}

function draw() {
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  design.elements.forEach(el => {
    const s = SCALE;
    const x=el.x*s, y=el.y*s, w=el.w*s, h=el.h*s;
    ctx.save();

    if (el.type === 'background') {
      const img = getImg(el);
      if (img) {
        const sc = Math.max(canvas.width/img.width, canvas.height/img.height);
        const dw=img.width*sc, dh=img.height*sc;
        ctx.drawImage(img, (canvas.width-dw)/2, (canvas.height-dh)/2, dw, dh);
      } else {
        ctx.fillStyle = el.color || '#1a1a2e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
    else if (el.type === 'photo') {
      const img = getImg(el);
      if (img) {
        const sc = h / img.height;
        const nw = img.width * sc;
        const sx = nw > w ? (nw-w)/2/sc : 0;
        const sw = nw > w ? w/sc : img.width;
        ctx.drawImage(img, sx, 0, sw, img.height, x, y, Math.min(nw,w), h);
      } else {
        ctx.fillStyle = 'rgba(80,80,160,0.25)';
        ctx.fillRect(x,y,w,h);
        ctx.strokeStyle='#3a3a80'; ctx.lineWidth=1; ctx.strokeRect(x,y,w,h);
        ctx.fillStyle='#556'; ctx.font=`${Math.min(h*.08,20)*1}px Inter`;
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText('👤 Ảnh minh họa', x+w/2, y+h/2);
      }
      // Badge: remove_bg indicator
      if (el.remove_bg) {
        const bw = 60*s, bh = 20*s, bx = x+4*s, by = y+4*s, br = 4*s;
        ctx.fillStyle = 'rgba(20,180,80,0.88)';
        ctx.beginPath(); ctx.roundRect(bx,by,bw,bh,br); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = `bold ${11*s}px Inter`;
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText('✂ Remove BG', bx+bw/2, by+bh/2);
      }
    }
    else if (el.type === 'text_main') {
      ctx.fillStyle='rgba(255,255,255,0.04)'; ctx.fillRect(x,y,w,h);
      ctx.strokeStyle='rgba(100,160,255,0.3)'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
      ctx.strokeRect(x,y,w,h); ctx.setLineDash([]);
      ctx.textBaseline='top';
      // Apply text transform
      let sampleText = el.sample_text || 'Text chính';
      const transform = el.text_transform || 'none';
      if (transform === 'upper') sampleText = sampleText.toUpperCase();
      else if (transform === 'lower') sampleText = sampleText.toLowerCase();
      const maxFs = Math.min((el.max_font||200) * s, h * 0.18);
      const minFs = Math.max((el.min_font||36) * s, 8);
      const align = el.align || 'center';
      const phrases = (el.sample_highlights||'').split('|').map(p=>p.trim().toLowerCase()).filter(p=>p);
      // Find best fit font size
      let bestFs = minFs, bestLines = [sampleText];
      for (let fs = maxFs; fs >= minFs; fs -= 1) {
        ctx.font = `bold ${fs}px Inter`;
        const lines = wrapCanvas(sampleText, w-20, fs);
        const lh = fs * 1.2 + (el.line_spacing||10)*s;
        const th = lines.length * lh;
        if (th <= h - 20) { bestFs = fs; bestLines = lines; break; }
      }
      ctx.font = `bold ${bestFs}px Inter`;
      const lh = bestFs * 1.2 + (el.line_spacing||10)*s;
      const th = bestLines.length * lh;
      let ty = y + Math.max(0, (h - th) / 2);
      const strokeW = (el.stroke_width||0) * s;
      const hiStrokeW = (el.hi_stroke_width||0) * s;
      bestLines.forEach(ln => {
        const lw = ctx.measureText(ln).width;
        let lx = align==='center' ? x+(w-lw)/2 : align==='right' ? x+w-lw-10 : x+10;
        const segs = phrases.length > 0 ? segmentHighlights(ln, phrases) : [{text:ln, hi:false}];
        ctx.textBaseline = 'top';
        let cx2 = lx;
        segs.forEach(({text:seg, hi}) => {
          const sw = hi ? hiStrokeW : strokeW;
          const sc = hi ? (el.hi_stroke_color||'#000000') : (el.stroke_color||'#000000');
          const fc = hi ? (el.highlight_color||'#ffca22') : (el.color||'#fff');
          if (sw > 0) {
            ctx.strokeStyle = sc; ctx.lineWidth = sw * 2; ctx.lineJoin = 'round';
            ctx.strokeText(seg, cx2, ty);
          }
          ctx.fillStyle = fc;
          ctx.fillText(seg, cx2, ty);
          cx2 += ctx.measureText(seg).width;
        });
        ty += lh;
      });
    }
    else if (el.type === 'hook') {
      ctx.fillStyle = el.bg_color||'#ffca22';
      if (el.shape==='pill') {
        const r = Math.min(h/2, 18*s);
        rrect(ctx, x, y, w, h, r); ctx.fill();
      } else if (el.shape!=='none') {
        ctx.fillRect(x,y,w,h);
      }
      ctx.fillStyle = el.color||'#000';
      const fixedSz = el.font_size||0;
      const hookText = (el.sample_text||'HOOK TEXT').toUpperCase();
      ctx.textAlign='center'; ctx.textBaseline='middle';
      if (fixedSz > 0) {
        const fs = fixedSz*s;
        ctx.font=`bold ${fs}px Inter`;
        if (ctx.measureText(hookText).width > w-40*s) {
          const lines = smartWrap2(hookText, w-40*s, fs);
          const lh = fs*1.15;
          const totalH = lines.length*lh;
          let ty = y+h/2 - totalH/2 + fs/2;
          lines.forEach(ln => { ctx.fillText(ln, x+w/2, ty); ty+=lh; });
        } else {
          ctx.fillText(hookText, x+w/2, y+h/2);
        }
      } else {
        // Auto-fit: largest size that fits max 2 lines
        let drawn = false;
        for (let sz=Math.floor(h/s); sz>=20; sz-=2) {
          const fs = sz*s;
          ctx.font=`bold ${fs}px Inter`;
          const tw = ctx.measureText(hookText).width;
          if (tw <= w-40*s) {
            ctx.fillText(hookText, x+w/2, y+h/2);
            drawn=true; break;
          } else {
            const lines = smartWrap2(hookText, w-40*s, fs);
            if (lines.length<=2) {
              const lh=fs*1.15, totalH=lines.length*lh;
              if (totalH <= h-10*s) {
                let ty=y+h/2-totalH/2+fs/2;
                lines.forEach(ln=>{ctx.fillText(ln,x+w/2,ty);ty+=lh;});
                drawn=true; break;
              }
            }
          }
        }
        if (!drawn) { const fs=20*s; ctx.font=`bold ${fs}px Inter`; ctx.fillText(hookText,x+w/2,y+h/2); }
      }
    }
    else if (el.type === 'text_box') {
      const rad = (el.border_radius||10)*s;
      ctx.fillStyle = el.bg_color||'#333333';
      if (rad > 0) { rrect(ctx, x, y, w, h, rad); ctx.fill(); }
      else { ctx.fillRect(x,y,w,h); }
      ctx.fillStyle = el.color||'#fff';
      const fs2 = (el.font_size||48)*s;
      ctx.font=`bold ${fs2}px Inter`; ctx.textAlign='center'; ctx.textBaseline='middle';
      const lines2 = wrapCanvas(el.sample_text||'', w-20*s, fs2);
      const lh2 = fs2*1.2;
      const totalH2 = lines2.length*lh2;
      let ty2 = y+h/2 - totalH2/2 + fs2/2;
      lines2.forEach(ln => { ctx.fillText(ln, x+w/2, ty2); ty2+=lh2; });
    }
    else if (el.type === 'text_static') {
      ctx.fillStyle='rgba(255,255,255,0.03)'; ctx.fillRect(x,y,w,h);
      ctx.strokeStyle='rgba(255,255,100,0.2)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.strokeRect(x,y,w,h); ctx.setLineDash([]);
      ctx.fillStyle=el.color||'#fff';
      const fs=(el.font_size||60)*s;
      ctx.font=`bold ${fs}px Inter`; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(el.text||'', x+w/2, y+h/2);
    }

    ctx.restore();

    if (el.id === selId) {
      ctx.strokeStyle='#6aacff'; ctx.lineWidth=2; ctx.setLineDash([]);
      ctx.strokeRect(x,y,w,h);
      handles(x,y,w,h);
    }
  });
}

function wrapCanvas(text, maxW, fs) {
  ctx.font = `bold ${fs}px Inter`;
  const words = text.split(' '), lines = []; let line = '';
  for (const w of words) {
    const t = line ? line+' '+w : w;
    if (ctx.measureText(t).width <= maxW) line=t;
    else { if(line) lines.push(line); line=w; }
  }
  if(line) lines.push(line);
  return lines.length ? lines : [text];
}

function smartWrap2(text, maxW, fs) {
  ctx.font = `bold ${fs}px Inter`;
  if (ctx.measureText(text).width <= maxW) return [text];
  let best = null;
  for (const sep of ['. ','! ','? ',', ','; ']) {
    let idx = -1;
    while ((idx = text.indexOf(sep, idx+1)) >= 0) {
      const l1 = text.slice(0, idx+sep.length).trim();
      const l2 = text.slice(idx+sep.length).trim();
      if (l1 && l2 && ctx.measureText(l1).width<=maxW && ctx.measureText(l2).width<=maxW)
        best = [l1, l2];
    }
  }
  if (best) return best;
  const words = text.split(' '), mid = text.length/2;
  let bestD = text.length, bestS = null, pos = 0;
  for (let i=0; i<words.length-1; i++) {
    pos += words[i].length + (i>0?1:0);
    const d = Math.abs(pos - mid);
    if (d < bestD) {
      const l1 = words.slice(0,i+1).join(' '), l2 = words.slice(i+1).join(' ');
      if (ctx.measureText(l1).width<=maxW && ctx.measureText(l2).width<=maxW)
        { bestD=d; bestS=[l1,l2]; }
    }
  }
  if (bestS) return bestS;
  return wrapCanvas(text, maxW, fs).slice(0,2);
}

function segmentHighlights(line, phrases) {
  // Find highlight spans in line text
  const lower = line.toLowerCase();
  const spans = [];
  phrases.forEach(ph => {
    let pos = 0;
    while (pos < lower.length) {
      const idx = lower.indexOf(ph, pos);
      if (idx === -1) break;
      spans.push([idx, idx + ph.length]);
      pos = idx + ph.length;
    }
  });
  spans.sort((a,b) => a[0]-b[0]);
  // Merge overlapping
  const merged = [];
  for (const [s,e] of spans) {
    if (merged.length && s <= merged[merged.length-1][1])
      merged[merged.length-1][1] = Math.max(e, merged[merged.length-1][1]);
    else merged.push([s,e]);
  }
  // Build segments
  const segs = []; let cur = 0;
  for (const [s,e] of merged) {
    if (s > cur) segs.push({text: line.slice(cur,s), hi: false});
    segs.push({text: line.slice(s,e), hi: true});
    cur = e;
  }
  if (cur < line.length) segs.push({text: line.slice(cur), hi: false});
  return segs.length ? segs : [{text: line, hi: false}];
}

function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r); ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h); ctx.arcTo(x,y+h,x,y+h-r,r);
  ctx.lineTo(x,y+r); ctx.arcTo(x,y,x+r,y,r);
  ctx.closePath();
}

const HS = 7;
function handlePts(x,y,w,h) {
  return [
    {cx:x,     cy:y,     h:'tl'}, {cx:x+w/2,cy:y,     h:'tc'}, {cx:x+w,   cy:y,     h:'tr'},
    {cx:x+w,   cy:y+h/2, h:'mr'}, {cx:x+w,   cy:y+h,  h:'br'}, {cx:x+w/2,cy:y+h,   h:'bc'},
    {cx:x,     cy:y+h,   h:'bl'}, {cx:x,     cy:y+h/2, h:'ml'},
  ];
}
function handles(x,y,w,h) {
  handlePts(x,y,w,h).forEach(({cx,cy}) => {
    ctx.fillStyle='#fff'; ctx.strokeStyle='#6aacff'; ctx.lineWidth=1.5;
    ctx.fillRect(cx-HS/2,cy-HS/2,HS,HS); ctx.strokeRect(cx-HS/2,cy-HS/2,HS,HS);
  });
}

// ========== MOUSE ==========
const CURSOR_MAP = {tl:'nw-resize',tc:'n-resize',tr:'ne-resize',mr:'e-resize',br:'se-resize',bc:'s-resize',bl:'sw-resize',ml:'w-resize'};

function pt(e) {
  const r = canvas.getBoundingClientRect();
  return {sx: e.clientX-r.left, sy: e.clientY-r.top,
          dx: (e.clientX-r.left)/SCALE, dy: (e.clientY-r.top)/SCALE};
}

function hitHandle(el, sx, sy) {
  const x=el.x*SCALE, y=el.y*SCALE, w=el.w*SCALE, h=el.h*SCALE;
  for (const {cx,cy,h:hh} of handlePts(x,y,w,h))
    if (Math.abs(sx-cx)<=HS+2 && Math.abs(sy-cy)<=HS+2) return hh;
  return null;
}
function hitEl(dx,dy) {
  for (let i=design.elements.length-1; i>=0; i--) {
    const e=design.elements[i];
    if (dx>=e.x&&dx<=e.x+e.w&&dy>=e.y&&dy<=e.y+e.h) return e;
  } return null;
}

function onDown(e) {
  canvas.focus();
  const {sx,sy,dx,dy} = pt(e);
  if (selId !== null) {
    const sel = design.elements.find(e=>e.id===selId);
    if (sel) {
      const h = hitHandle(sel, sx, sy);
      if (h) {
        drag = {type:'resize',h,sx,sy,ox:sel.x,oy:sel.y,ow:sel.w,oh:sel.h,el:sel};
        return;
      }
    }
  }
  const el = hitEl(dx,dy);
  if (el) {
    selId = el.id;
    drag = {type:'move',dx,dy,ox:el.x,oy:el.y,el};
    draw(); renderList(); renderProps();
  } else {
    selId = null; drag = null;
    draw(); renderList(); renderProps();
  }
}

function onMove(e) {
  const {sx,sy,dx,dy} = pt(e);
  if (!drag) {
    if (selId !== null) {
      const sel = design.elements.find(e=>e.id===selId);
      if (sel) { const h=hitHandle(sel,sx,sy); canvas.style.cursor=h?CURSOR_MAP[h]:'move'; return; }
    }
    canvas.style.cursor = hitEl(dx,dy) ? 'move' : 'default';
    return;
  }
  const ddx=(sx-drag.sx)/SCALE, ddy=(sy-drag.sy)/SCALE;
  const el=drag.el;
  if (drag.type==='move') {
    el.x = Math.max(0, Math.round(drag.ox+dx-drag.dx));
    el.y = Math.max(0, Math.round(drag.oy+dy-drag.dy));
  } else {
    let nx=drag.ox, ny=drag.oy, nw=drag.ow, nh=drag.oh;
    const h=drag.h;
    if (h.includes('r')) nw = Math.max(40, Math.round(drag.ow+ddx));
    if (h.includes('l')) { nx=Math.round(drag.ox+ddx); nw=Math.max(40,Math.round(drag.ow-ddx)); }
    if (h.includes('b')) nh = Math.max(20, Math.round(drag.oh+ddy));
    if (h.includes('t')) { ny=Math.round(drag.oy+ddy); nh=Math.max(20,Math.round(drag.oh-ddy)); }
    el.x=nx; el.y=ny; el.w=nw; el.h=nh;
  }
  draw();
  syncXYWH();
}

function onUp() { drag=null; renderProps(); }

function onDbl(e) {
  const {dx,dy} = pt(e);
  const el = hitEl(dx,dy);
  if (!el) return;
  const field = el.type==='text_static'?'text':'sample_text';
  if (field in el) {
    const v = prompt('Nhập text:', el[field]||'');
    if (v!==null) { el[field]=v; draw(); renderProps(); }
  }
}

function onKey(e) {
  if (!selId) return;
  if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
  const el = design.elements.find(x=>x.id===selId);
  if (!el) return;
  const step = e.shiftKey?10:1;
  if (e.key==='ArrowLeft')  { el.x-=step; e.preventDefault(); }
  if (e.key==='ArrowRight') { el.x+=step; e.preventDefault(); }
  if (e.key==='ArrowUp')    { el.y-=step; e.preventDefault(); }
  if (e.key==='ArrowDown')  { el.y+=step; e.preventDefault(); }
  if (e.key==='Delete')     { delEl(selId); return; }
  draw(); syncXYWH();
}

// ========== ELEMENT LIST ==========
function renderList() {
  const list = document.getElementById('element-list');
  list.innerHTML = '';
  [...design.elements].reverse().forEach(el => {
    const d = document.createElement('div');
    d.className = 'el-item' + (el.id===selId?' selected':'');
    d.innerHTML = `<span class="el-name">${EL_ICON[el.type]||'?'} ${EL_LABEL[el.type]||el.type}</span>
      <span class="el-btn" onclick="event.stopPropagation();moveDn(${el.id})" title="Lên">▲</span>
      <span class="el-btn" onclick="event.stopPropagation();moveUp(${el.id})" title="Xuống">▼</span>
      <span class="el-btn" style="color:#c44" onclick="event.stopPropagation();delEl(${el.id})" title="Xóa">✕</span>`;
    d.onclick = () => selEl(el.id);
    list.appendChild(d);
  });
}

// ========== PROPERTIES ==========
function renderProps() {
  const el = design.elements.find(e=>e.id===selId);
  const p = document.getElementById('props');
  if (!el) { p.innerHTML='<div id="no-sel">Click vào phần tử để chỉnh sửa</div>'; return; }
  p.innerHTML = buildProps(el);
}

function fopt(list, val) {
  return list.map(f=>`<option value="${f}"${f===val?' selected':''}>${f}</option>`).join('');
}
function aopt(val) {
  return ['left','center','right'].map(a=>`<option value="${a}"${a===val?' selected':''}>${a}</option>`).join('');
}
function shopt(val) {
  return ['pill','rect','none'].map(s=>`<option value="${s}"${s===val?' selected':''}>${s}</option>`).join('');
}

function buildProps(el) {
  const pos = `<div class="pg"><label>Vị trí & Kích thước</label>
    <div class="row4">
      <div><input type="number" id="px" value="${el.x}" onchange="sp('x',+this.value)"><div class="lbl">X</div></div>
      <div><input type="number" id="py" value="${el.y}" onchange="sp('y',+this.value)"><div class="lbl">Y</div></div>
      <div><input type="number" id="pw" value="${el.w}" onchange="sp('w',+this.value)"><div class="lbl">W</div></div>
      <div><input type="number" id="ph" value="${el.h}" onchange="sp('h',+this.value)"><div class="lbl">H</div></div>
    </div></div>`;

  let spec = '';

  if (el.type==='background') {
    spec = `
    <div class="pg"><label>Màu nền (fallback)</label>
      <input type="color" value="${el.color||'#1a1a2e'}" oninput="sp('color',this.value)">
    </div>
    <div class="pg"><label>Ảnh nền</label>
      <button class="btn" style="width:100%" onclick="document.getElementById('fi-bg').click()">📁 Chọn ảnh nền</button>
      ${el.image_path?`<div style="font-size:10px;color:#666;margin-top:5px;word-break:break-all">${el.image_path}</div>`:''}
    </div>`;
  }
  else if (el.type==='photo') {
    spec = `
    <div class="pg"><label>Preview ảnh (tạm, không lưu vào py)</label>
      <button class="btn" style="width:100%" onclick="document.getElementById('fi-photo').click()">📁 Chọn ảnh preview</button>
      <div style="font-size:10px;color:#666;margin-top:5px">Ảnh thật sẽ lấy từ<br>VISUAL/{code}/thumbnail/thumb_004.png</div>
    </div>
    <div class="pg">
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" ${el.remove_bg?'checked':''} onchange="sp('remove_bg',this.checked);draw()" style="width:16px;height:16px">
        <span>✂️ Tách nền (Remove BG)</span>
      </label>
      <div style="font-size:10px;color:#888;margin-top:4px">Dùng rembg hoặc MODNet để xóa nền ảnh</div>
    </div>`;
  }
  else if (el.type==='text_main') {
    const ttLabels = {'none':'Giữ nguyên','upper':'IN HOA','lower':'in thường'};
    const ttOpts = ['none','upper','lower'].map(v=>`<option value="${v}"${(el.text_transform||'none')===v?' selected':''}>${ttLabels[v]}</option>`).join('');
    spec = `
    <div class="pg"><label>Text mẫu (để preview)</label>
      <textarea oninput="sp('sample_text',this.value)">${el.sample_text||''}</textarea>
    </div>
    <div class="pg">
      <label>Highlights (phân cách bằng |)</label>
      <textarea oninput="sp('sample_highlights',this.value)" placeholder="VD: từ nổi bật|cụm khác">${el.sample_highlights||''}</textarea>
      <div style="font-size:10px;color:#555;margin-top:3px">Các đoạn này sẽ đổi màu thành màu highlight</div>
    </div>
    <div class="pg"><label>Font</label>
      <select onchange="sp('font',this.value)">${fopt(FONTS,el.font)}</select>
    </div>
    <div class="pg"><label>Kiểu chữ (transform)</label>
      <select onchange="sp('text_transform',this.value)">${ttOpts}</select>
    </div>
    <div class="pg"><label>Màu chữ</label>
      <input type="color" value="${el.color||'#ffffff'}" oninput="sp('color',this.value)">
    </div>
    <div class="pg"><label>Viền chữ thường</label>
      <div class="row2">
        <div><input type="number" value="${el.stroke_width||0}" min="0" onchange="sp('stroke_width',+this.value)"><div class="lbl" style="font-size:9px;color:#666;text-align:center">Width</div></div>
        <div><input type="color" value="${el.stroke_color||'#000000'}" oninput="sp('stroke_color',this.value)"></div>
      </div>
    </div>
    <div class="pg"><label>Màu highlight</label>
      <input type="color" value="${el.highlight_color||'#ffca22'}" oninput="sp('highlight_color',this.value)">
    </div>
    <div class="pg"><label>Viền chữ highlight</label>
      <div class="row2">
        <div><input type="number" value="${el.hi_stroke_width||0}" min="0" onchange="sp('hi_stroke_width',+this.value)"><div class="lbl" style="font-size:9px;color:#666;text-align:center">Width</div></div>
        <div><input type="color" value="${el.hi_stroke_color||'#000000'}" oninput="sp('hi_stroke_color',this.value)"></div>
      </div>
    </div>
    <div class="pg"><label>Căn chỉnh</label>
      <select onchange="sp('align',this.value)">${aopt(el.align)}</select>
    </div>
    <div class="pg"><label>Font size auto-fit</label>
      <div class="row2">
        <div><input type="number" value="${el.min_font||36}" placeholder="Min" onchange="sp('min_font',+this.value)"><div class="lbl" style="font-size:9px;color:#666;text-align:center">Min</div></div>
        <div><input type="number" value="${el.max_font||200}" placeholder="Max" onchange="sp('max_font',+this.value)"><div class="lbl" style="font-size:9px;color:#666;text-align:center">Max</div></div>
      </div>
      <input type="number" value="${el.line_spacing||10}" placeholder="Line spacing" onchange="sp('line_spacing',+this.value)" style="margin-top:4px">
      <div class="lbl" style="font-size:9px;color:#666;margin-top:2px">Line spacing (px)</div>
    </div>`;
  }
  else if (el.type==='hook') {
    spec = `
    <div class="pg"><label>Text mẫu</label>
      <input type="text" value="${el.sample_text||''}" onchange="sp('sample_text',this.value);draw()">
    </div>
    <div class="pg"><label>Font</label>
      <select onchange="sp('font',this.value);draw()">${fopt(FONTS,el.font)}</select>
    </div>
    <div class="pg"><label>Cỡ chữ (0 = tự động)</label>
      <input type="number" value="${el.font_size||0}" min="0" placeholder="0 = auto-fit" onchange="sp('font_size',+this.value);draw()">
      <div class="lbl" style="font-size:9px;color:#666;margin-top:2px">Nếu chữ dài sẽ tự xuống 2 dòng</div>
    </div>
    <div class="pg"><label>Màu chữ</label>
      <input type="color" value="${el.color||'#000000'}" oninput="sp('color',this.value)">
    </div>
    <div class="pg"><label>Màu nền bar</label>
      <input type="color" value="${el.bg_color||'#ffca22'}" oninput="sp('bg_color',this.value)">
    </div>
    <div class="pg"><label>Hình dạng</label>
      <select onchange="sp('shape',this.value);draw()">${shopt(el.shape)}</select>
    </div>
    <div class="pg"><label>Stroke chữ</label>
      <div class="row2">
        <div><input type="number" value="${el.stroke_width||0}" min="0" onchange="sp('stroke_width',+this.value)"><div class="lbl" style="font-size:9px;color:#666;text-align:center">Width</div></div>
        <div><input type="color" value="${el.stroke_color||'#000000'}" oninput="sp('stroke_color',this.value)"></div>
      </div>
    </div>`;
  }
  else if (el.type==='text_box') {
    spec = `
    <div class="pg"><label>Nội dung</label>
      <input type="text" value="${el.sample_text||''}" onchange="sp('sample_text',this.value);draw()">
    </div>
    <div class="pg"><label>Font</label>
      <select onchange="sp('font',this.value);draw()">${fopt(FONTS,el.font)}</select>
    </div>
    <div class="pg"><label>Cỡ chữ (px)</label>
      <input type="number" value="${el.font_size||48}" onchange="sp('font_size',+this.value);draw()">
    </div>
    <div class="pg"><label>Màu chữ</label>
      <input type="color" value="${el.color||'#ffffff'}" oninput="sp('color',this.value)">
    </div>
    <div class="pg"><label>Màu nền khung</label>
      <input type="color" value="${el.bg_color||'#333333'}" oninput="sp('bg_color',this.value)">
    </div>
    <div class="pg"><label>Bo góc (px)</label>
      <input type="number" value="${el.border_radius||10}" min="0" onchange="sp('border_radius',+this.value);draw()">
    </div>
    <div class="pg"><label>Căn chỉnh</label>
      <select onchange="sp('align',this.value);draw()">${aopt(el.align)}</select>
    </div>`;
  }
  else if (el.type==='text_static') {
    spec = `
    <div class="pg"><label>Nội dung</label>
      <input type="text" value="${el.text||''}" onchange="sp('text',this.value);draw()">
    </div>
    <div class="pg"><label>Font</label>
      <select onchange="sp('font',this.value);draw()">${fopt(FONTS,el.font)}</select>
    </div>
    <div class="pg"><label>Cỡ chữ (px)</label>
      <input type="number" value="${el.font_size||60}" onchange="sp('font_size',+this.value);draw()">
    </div>
    <div class="pg"><label>Màu chữ</label>
      <input type="color" value="${el.color||'#ffffff'}" oninput="sp('color',this.value)">
    </div>
    <div class="pg"><label>Căn chỉnh</label>
      <select onchange="sp('align',this.value)">${aopt(el.align)}</select>
    </div>`;
  }

  return pos + spec;
}

function sp(key, val) {
  const el = design.elements.find(e=>e.id===selId);
  if (!el) return;
  el[key] = val;
  draw();
}

function syncXYWH() {
  const el = design.elements.find(e=>e.id===selId);
  if (!el) return;
  ['px','py','pw','ph'].forEach((id,i) => {
    const inp = document.getElementById(id);
    if (inp) inp.value = Math.round([el.x,el.y,el.w,el.h][i]);
  });
}

// ========== UPLOAD ==========
async function onBgUpload(e) {
  const file = e.target.files[0]; if (!file) return;
  const tname = document.getElementById('tname').value.trim() || 'untitled';
  const fd = new FormData();
  fd.append('file', file); fd.append('save_as', `bg:${tname}`);
  status('Đang upload background...');
  const res = await fetch('/upload-image', {method:'POST', body:fd});
  const data = await res.json();
  if (data.url) {
    const el = design.elements.find(e=>e.id===selId && e.type==='background');
    if (el) { el.src=data.url; el.image_path=data.asset_relative||''; draw(); renderProps(); }
  }
  status('Background đã lưu: ' + (data.asset_relative||''));
  e.target.value='';
}

async function onPhotoUpload(e) {
  const file = e.target.files[0]; if (!file) return;
  const fd = new FormData();
  fd.append('file', file); fd.append('save_as', 'preview');
  const res = await fetch('/upload-image', {method:'POST', body:fd});
  const data = await res.json();
  if (data.url) {
    const el = design.elements.find(e=>e.id===selId && e.type==='photo');
    if (el) { el.src=data.url; draw(); }
  }
  e.target.value='';
}

// ========== PREVIEW ==========
async function doPreview() {
  status('Đang render PIL preview...');
  const res = await fetch('/preview', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(exportDesign())
  });
  const data = await res.json();
  if (data.image) {
    document.getElementById('prev-img').src = data.image;
    document.getElementById('overlay').classList.add('show');
    status('Preview done');
  } else {
    status('Lỗi: ' + (data.error||'unknown'));
    alert('Preview lỗi:\n' + (data.error||'') + '\n\n' + (data.trace||''));
  }
}

// ========== EXPORT ==========
async function doExport() {
  const tname = document.getElementById('tname').value.trim();
  if (!tname) { alert('Nhập tên template trước (vd: KA1-T5)'); return; }
  status('Đang tạo THUMB_' + tname + '.py...');
  const res = await fetch('/export', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({design: exportDesign(), template_name: tname})
  });
  const data = await res.json();
  if (data.success) {
    showBanner('✓ Đã lưu THUMB_' + tname + '.py');
    status('Xuất file: ' + data.path);
  } else {
    status('Lỗi: ' + (data.error||'unknown'));
    alert('Export lỗi:\n' + (data.error||'') + '\n\n' + (data.trace||''));
  }
}

function exportDesign() {
  const thumbOrder = document.getElementById('thumb_order').value;
  return {
    canvas: design.canvas,
    thumb_order: thumbOrder,
    elements: design.elements.map(el => {
      const e = Object.assign({}, el);
      delete e.src;  // Remove blob URL
      return e;
    })
  };
}

// ========== SAVE / LOAD JSON ==========
function saveJSON() {
  const tname = document.getElementById('tname').value.trim() || 'design';
  const blob = new Blob([JSON.stringify({template_name:tname, design}, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `THUMB_${tname}.json`;
  a.click();
  status('Đã save JSON');
}

function loadJSON() {
  const inp = document.createElement('input');
  inp.type='file'; inp.accept='.json';
  inp.onchange = e => {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const d = JSON.parse(ev.target.result);
        if (d.template_name) document.getElementById('tname').value = d.template_name;
        if (d.design) {
          design = d.design;
          uid = Math.max(...design.elements.map(e=>e.id||0), 0);
          selId = null;
          draw(); renderList(); renderProps();
          status('Đã load design');
        }
      } catch(err) { alert('File JSON không hợp lệ: ' + err); }
    };
    reader.readAsText(file);
  };
  inp.click();
}

function loadPY() {
  const inp = document.createElement('input');
  inp.type = 'file'; inp.accept = '.py';
  inp.onchange = e => {
    const file = e.target.files[0]; if (!file) return;
    // Extract template name from filename: THUMB_KA1-T3.py → KA1-T3
    const tmatch = file.name.match(/THUMB_(.+)\.py$/i);
    const tname = tmatch ? tmatch[1] : file.name.replace(/\.py$/i,'');
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const content = ev.target.result;
        // DESIGN block is between "DESIGN = {" and the next "# ===" line
        const match = content.match(/^DESIGN\s*=\s*(\{[\s\S]*?)\n# ===/m);
        if (!match) {
          alert('Không tìm thấy DESIGN block.\nChỉ hỗ trợ file được Export từ Thumb Designer.');
          return;
        }
        let raw = match[1];
        raw = raw.split('True').join('true');
        raw = raw.split('False').join('false');
        raw = raw.split('None').join('null');
        const d = JSON.parse(raw);
        if (!d.canvas || !d.elements) {
          alert('DESIGN không đúng định dạng (cần canvas + elements).');
          return;
        }
        document.getElementById('tname').value = tname;
        design = d;
        uid = Math.max(...design.elements.map(el => el.id || 0), 0);
        selId = null;
        draw(); renderList(); renderProps();
        status('Đã load: ' + file.name + ' → ' + tname);
      } catch(err) { alert('Lỗi parse file .py:\n' + err); }
    };
    reader.readAsText(file, 'utf-8');
  };
  inp.click();
}

// ========== UTILS ==========
function status(msg) { document.getElementById('statusbar').textContent = msg; }
function showBanner(msg) {
  const b = document.getElementById('msg-banner');
  b.textContent = msg; b.style.display='block';
  setTimeout(()=>b.style.display='none', 3500);
}
</script>
</body>
</html>"""


# ============ MAIN ============
def main():
    def open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open(f'http://localhost:{PORT}')

    print(f"\n{'='*52}")
    print(f"  VE3 Thumb Designer")
    print(f"  URL: http://localhost:{PORT}")
    print(f"  Ctrl+C để thoát")
    print(f"{'='*52}\n")

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
