#!/usr/bin/env python3
"""
EchidnaComicEngine Pro
======================
Autonomous End-to-End AI Comic & Manga Localization Engine.
Engineered for:
  - 100% Pure White Speech Bubble Cleaning (No Telea Gray Smudges)
  - Ghost-Free Inpainting of Artwork Sound Effects (SFX) & Text
  - Morphological Letter-Bridging Moment Centroid Placement
  - Dynamic Diamond/Oval Silhouette Typesetting
  - Style-Matched Typography for Dialogue, Screams, Narration & Action SFX
  - Automated Batch CBZ Processing with Automatic Retry & Zero Dropped Pages
"""

import os
import sys
import io
import re
import json
import time
import base64
import zipfile
import textwrap
import argparse
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

import arabic_reshaper
from bidi.algorithm import get_display

API_ENDPOINT = "http://localhost:20128/v1/chat/completions"
DEFAULT_MODEL = "ag/gemini-3.8-flash-low"
FALLBACK_MODEL = "ag/gemini-3.7-flash-low"
RETRY_MODEL = "ag/gemini-3.8-flash"

FONT_BOLD = r"C:\Users\ITcenter\Downloads\vazir-font-extracted\Vazir-Bold.ttf"
FONT_BLACK = r"C:\Users\ITcenter\Downloads\vazir-font-extracted\Vazir-Black.ttf"
FONT_MEDIUM = r"C:\Users\ITcenter\Downloads\vazir-font-extracted\Vazir-Medium.ttf"

for f, alt in [(FONT_BOLD, r"C:\Users\ITcenter\Downloads\Vazir-Bold.ttf"),
              (FONT_BLACK, r"C:\Users\ITcenter\AppData\Local\Microsoft\Windows\Fonts\Vazir-Black.ttf"),
              (FONT_MEDIUM, r"C:\Users\ITcenter\AppData\Local\Microsoft\Windows\Fonts\Vazir-Medium.ttf")]:
    if not os.path.exists(f) and os.path.exists(alt):
        if "Bold" in f: FONT_BOLD = alt
        elif "Black" in f: FONT_BLACK = alt
        elif "Medium" in f: FONT_MEDIUM = alt


def get_api_key() -> str:
    key = os.environ.get("HERMES_CUSTOM_LOCALHOST_20128_API_KEY", "")
    if not key:
        env_file = r"C:\Users\ITcenter\AppData\Local\hermes\.env"
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if "HERMES_CUSTOM_LOCALHOST_20128_API_KEY" in line:
                            key = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass
    return key


PROMPT_TEMPLATE = """You are an elite master comic scanlation specialist.
Analyze this comic page and extract EVERY piece of text:
1. Speech bubbles (dialogue)
2. Shout/Scream bubbles (jagged/explosive bubbles)
3. Narration boxes (rectangular captions)
4. Sound effects (SFX / onomatopoeia written on art or background)
5. Background signs, titles, screens, and written text

For each text occurrence, provide:
- "id": integer starting from 1 in natural reading order
- "type": "bubble" | "scream" | "narration" | "sfx" | "background"
- "box_1000": [ymin, xmin, ymax, xmax] (normalized integers 0-1000 tightly bounding the text)
- "translation_fa": culturally authentic, colloquial Iranian Persian translation.
  CRITICAL TRANSLATION RULES:
  * Rick: cynical, aggressive, uses harsh Iranian street slang, insults, and profanity without censorship.
  * Morty: anxious, stammering, hesitant, uses stuttering filler marks ("مـ... من...", "آخه...").
  * Jerry: pathetic, insecure, defensive.
  * SFX: translate into punchy, evocative Persian comic sounds ("شترررق!", "زووووییی!", "بوووم!", "هوووورت!", "کـِـراااک!").
  * DO NOT censor any profanities or curses.

Return ONLY a valid JSON array of objects. No markdown chatter."""


def extract_page_data_ai(image_path: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    img = Image.open(image_path)
    orig_w, orig_h = img.size

    scale = 1024 / max(orig_w, orig_h)
    prev_w, prev_h = int(orig_w * scale), int(orig_h * scale)
    preview = img.resize((prev_w, prev_h), Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    api_key = get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    models_to_try = [DEFAULT_MODEL, FALLBACK_MODEL, RETRY_MODEL]
    last_err = None

    for attempt in range(max_retries):
        cur_model = models_to_try[attempt % len(models_to_try)]
        payload = {
            "model": cur_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEMPLATE},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "stream": False
        }

        try:
            resp = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=40)
            if resp.status_code != 200:
                time.sleep(1.5)
                continue

            content = resp.json()["choices"][0]["message"]["content"].strip()
            if not content:
                time.sleep(1.5)
                continue

            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:].strip()
                content = content.strip()

            try:
                items = json.loads(content)
                if isinstance(items, list):
                    return items
            except Exception:
                match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
                if match:
                    items = json.loads(match.group(0))
                    if isinstance(items, list):
                        return items
        except Exception as e:
            last_err = e
            time.sleep(1.5)

    raise ValueError(f"Failed to extract page data after {max_retries} attempts: {last_err}")


def clean_crop_region(crop: np.ndarray, item_type: str = "bubble") -> np.ndarray:
    """
    Cleans text from crop region with 100% zero-ghosting:
    - Bubbles: Pure solid interior restoration (no Telea gray blur).
    - Narration: Solid color fill of background.
    - SFX & Artwork: Dilated contrast mask inpainting to eliminate black outlines.
    """
    h_c, w_c = crop.shape[:2]
    if h_c < 10 or w_c < 10:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    median_val = np.median(gray)

    if item_type in ["bubble", "narration"]:
        if median_val > 155:
            # White / bright speech bubble: text ink is dark
            dark_thresh = min(160, int(median_val * 0.80))
            dark = (gray < dark_thresh).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
            
            mask = np.zeros_like(gray)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                # Exclude outer border touching all edges
                if w > w_c * 0.96 or h > h_c * 0.96:
                    continue
                if area < (h_c * w_c * 0.65):
                    mask[labels == i] = 255

            # Dilate mask to absorb anti-aliased font edges
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(mask, k, iterations=1)

            # Paint with pure white / interior color (no telea blur)
            cleaned = crop.copy()
            cleaned[dilated > 0] = [255, 255, 255]
            return cleaned
        else:
            # Dark / shout bubble: text is bright
            bright_thresh = max(110, int(median_val * 1.35))
            bright = (gray > bright_thresh).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
            
            mask = np.zeros_like(gray)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                if w > w_c * 0.96 or h > h_c * 0.96:
                    continue
                if area < (h_c * w_c * 0.65):
                    mask[labels == i] = 255

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(mask, k, iterations=1)
            cleaned = crop.copy()
            cleaned[dilated > 0] = [10, 10, 10]
            return cleaned
    else:
        # SFX, scream, signs, or artwork text
        diff_from_median = np.abs(gray.astype(np.int32) - int(median_val)).astype(np.uint8)
        _, contrast_mask = cv2.threshold(diff_from_median, 30, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(contrast_mask)
        clean_mask = np.zeros_like(gray)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if x > 1 and y > 1 and (x + w) < (w_c - 1) and (y + h) < (h_c - 1) and area < (h_c * w_c * 0.70):
                clean_mask[labels == i] = 255

        # Generous dilation to swallow outer stroke lines and drop shadows
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(clean_mask, k, iterations=2)
        return cv2.inpaint(crop, dilated, 5, cv2.INPAINT_TELEA)


def compute_true_centroid(crop: np.ndarray, is_bubble: bool = True) -> Tuple[float, float, int, int]:
    """Calculates mathematical centroid and safe interior dimensions via letter-bridged image moments."""
    h_c, w_c = crop.shape[:2]
    default_cx, default_cy = w_c / 2.0, h_c / 2.0
    safe_w, safe_h = int(w_c * 0.85), int(h_c * 0.85)

    if not is_bubble or h_c < 20 or w_c < 20:
        return default_cx, default_cy, safe_w, safe_h

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    median_val = np.median(gray)
    white = (gray > 175).astype(np.uint8) * 255 if median_val > 155 else (gray < 85).astype(np.uint8) * 255

    # Letter bridging: bridge black letters so bubble interior forms one unified component
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    cy_c, cx_c = h_c // 2, w_c // 2
    best_id = labels[cy_c, cx_c]
    if best_id == 0 or stats[best_id, cv2.CC_STAT_AREA] < 300:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_id = np.argmax(areas) + 1 if len(areas) > 0 else 0

    if best_id == 0 or stats[best_id, cv2.CC_STAT_AREA] < 300:
        return default_cx, default_cy, safe_w, safe_h

    bubble_mask = (labels == best_id).astype(np.uint8) * 255
    bw_box = stats[best_id, cv2.CC_STAT_WIDTH]
    bh_box = stats[best_id, cv2.CC_STAT_HEIGHT]

    # Erode tail
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    core = cv2.erode(bubble_mask, k_erode, iterations=1)
    if np.sum(core) == 0:
        core = bubble_mask

    M = cv2.moments(core)
    if M["m00"] == 0:
        return default_cx, default_cy, int(bw_box * 0.85), int(bh_box * 0.85)

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy, int(bw_box * 0.88), int(bh_box * 0.88)


def format_persian_lines(text: str, font_path: str, max_w: int, max_h: int, is_oval: bool = True) -> Tuple[List[str], int]:
    """Dynamically calculates font size and diamond/oval silhouette wraps."""
    words = text.split()
    if not words:
        return [text], 20

    for fsize in range(40, 11, -2):
        font = ImageFont.truetype(font_path, fsize)
        step = int(fsize * 1.20)
        total_words = len(words)

        for num_lines in range(1, min(6, total_words + 1)):
            avg_chars = max(10, len(text) // num_lines + 3)
            lines = textwrap.wrap(text, width=avg_chars)
            tot_h = len(lines) * step
            if tot_h > max_h:
                continue

            fits = True
            for i, l in enumerate(lines):
                if is_oval and len(lines) > 2 and (i == 0 or i == len(lines) - 1):
                    allowed_w = max_w * 0.76
                else:
                    allowed_w = max_w * 0.95

                reshaped = get_display(arabic_reshaper.reshape(l))
                bbox = font.getbbox(reshaped)
                lw = bbox[2] - bbox[0]
                if lw > allowed_w:
                    fits = False
                    break
            if fits:
                return lines, fsize

    min_size = 12
    return textwrap.wrap(text, width=22), min_size


def render_page(image_cv: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
    """Master rendering pipeline: cleaning, centroid moments, silhouette typesetting."""
    H, W = image_cv.shape[:2]
    clean_img = image_cv.copy()

    # Step 1: Clean all text regions
    for it in items:
        box = it.get("box_1000", [0, 0, 1000, 1000])
        ymin, xmin, ymax, xmax = box
        pad = 8
        y1 = max(0, int(ymin * H / 1000.0) - pad)
        y2 = min(H, int(ymax * H / 1000.0) + pad)
        x1 = max(0, int(xmin * W / 1000.0) - pad)
        x2 = min(W, int(xmax * W / 1000.0) + pad)

        crop = clean_img[y1:y2, x1:x2]
        item_type = it.get("type", "bubble")
        clean_img[y1:y2, x1:x2] = clean_crop_region(crop, item_type=item_type)

    # Step 2: Typeset dialogue and sound effects
    pil_img = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for it in items:
        box = it.get("box_1000", [0, 0, 1000, 1000])
        ymin, xmin, ymax, xmax = box
        y1 = int(ymin * H / 1000.0)
        y2 = int(ymax * H / 1000.0)
        x1 = int(xmin * W / 1000.0)
        x2 = int(xmax * W / 1000.0)

        item_type = it.get("type", "bubble")
        is_bubble = item_type in ["bubble", "scream", "narration"]
        crop = clean_img[y1:y2, x1:x2]
        cx_rel, cy_rel, safe_w, safe_h = compute_true_centroid(crop, is_bubble=is_bubble)
        abs_cx = x1 + cx_rel
        abs_cy = y1 + cy_rel

        fa_text = it.get("translation_fa", "").strip()
        if not fa_text:
            continue

        if item_type == "sfx":
            # Dynamic sound effect
            bw = x2 - x1
            bh = y2 - y1
            shaped = get_display(arabic_reshaper.reshape(fa_text))
            fsize = max(24, min(54, int(min(bw, bh) * 0.45)))
            font = ImageFont.truetype(FONT_BLACK, fsize)
            bbox = font.getbbox(shaped)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # If vertical effect (height > 1.4 * width), render along vertical action axis
            if bh > 1.4 * bw and len(fa_text) > 2:
                txt_canvas = Image.new("RGBA", (int(tw + 40), int(th + 40)), (0, 0, 0, 0))
                d_cv = ImageDraw.Draw(txt_canvas)
                # Drop shadow + heavy outline
                d_cv.text((23, 23), shaped, font=font, fill=(0, 0, 0, 220), stroke_width=6, stroke_fill=(0, 0, 0, 255))
                d_cv.text((20, 20), shaped, font=font, fill=(240, 25, 35, 255), stroke_width=5, stroke_fill=(0, 0, 0, 255))
                rotated = txt_canvas.rotate(-75, expand=True, resample=Image.Resampling.BICUBIC)
                rx = int(abs_cx - rotated.width / 2)
                ry = int(abs_cy - rotated.height / 2)
                pil_img.paste(rotated, (rx, ry), rotated)
            else:
                lx = abs_cx - tw / 2.0
                ly = abs_cy - th / 2.0
                # Drop shadow
                draw.text((lx + 3, ly + 3), shaped, font=font, fill=(0, 0, 0), stroke_width=5, stroke_fill=(0, 0, 0))
                # Vibrant red action fill
                draw.text((lx, ly), shaped, font=font, fill=(240, 25, 35), stroke_width=4, stroke_fill=(0, 0, 0))
        elif item_type == "scream":
            # Heavy screaming bubble
            lines, fsize = format_persian_lines(fa_text, FONT_BLACK, safe_w, safe_h, is_oval=True)
            font = ImageFont.truetype(FONT_BLACK, fsize)
            step = int(fsize * 1.20)
            tot_h = len(lines) * step
            curr_y = abs_cy - (tot_h / 2.0)
            for line in lines:
                shaped_line = get_display(arabic_reshaper.reshape(line))
                bbox = font.getbbox(shaped_line)
                lw = bbox[2] - bbox[0]
                lx = abs_cx - (lw / 2.0)
                draw.text((lx, curr_y), shaped_line, font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(0, 0, 0))
                curr_y += step
        elif item_type == "narration":
            # Narration box (rectangular)
            lines, fsize = format_persian_lines(fa_text, FONT_MEDIUM, safe_w, safe_h, is_oval=False)
            font = ImageFont.truetype(FONT_MEDIUM, fsize)
            step = int(fsize * 1.20)
            tot_h = len(lines) * step
            curr_y = abs_cy - (tot_h / 2.0)
            for line in lines:
                shaped_line = get_display(arabic_reshaper.reshape(line))
                bbox = font.getbbox(shaped_line)
                lw = bbox[2] - bbox[0]
                lx = abs_cx - (lw / 2.0)
                draw.text((lx, curr_y), shaped_line, font=font, fill=(15, 15, 15))
                curr_y += step
        else:
            # Standard speech bubble
            lines, fsize = format_persian_lines(fa_text, FONT_BOLD, safe_w, safe_h, is_oval=True)
            font = ImageFont.truetype(FONT_BOLD, fsize)
            step = int(fsize * 1.20)
            tot_h = len(lines) * step
            curr_y = abs_cy - (tot_h / 2.0)
            for line in lines:
                shaped_line = get_display(arabic_reshaper.reshape(line))
                bbox = font.getbbox(shaped_line)
                lw = bbox[2] - bbox[0]
                lx = abs_cx - (lw / 2.0)
                draw.text((lx, curr_y), shaped_line, font=font, fill=(10, 10, 10), stroke_width=1, stroke_fill=(10, 10, 10))
                curr_y += step

    rendered_rgb = np.array(pil_img)
    return cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)


def process_single_image(input_path: str, output_path: str) -> None:
    t0 = time.time()
    items = extract_page_data_ai(input_path)
    raw_img = cv2.imread(input_path)
    if raw_img is None:
        raise ValueError(f"Could not load image: {input_path}")
    out_img = render_page(raw_img, items)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 93])


def translate_full_archive(cbz_path: str, out_cbz: str, workdir: str) -> None:
    raw_dir = os.path.join(workdir, "raw")
    trans_dir = os.path.join(workdir, "translated")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    with zipfile.ZipFile(cbz_path, "r") as z:
        files = sorted([f for f in z.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
        total = len(files)
        print(f"[*] Starting full translation of {total} pages...")

        for idx, file_name in enumerate(files, 1):
            base_bname = os.path.basename(file_name)
            raw_target = os.path.join(raw_dir, base_bname)
            out_target = os.path.join(trans_dir, base_bname)

            if not os.path.exists(raw_target):
                with open(raw_target, "wb") as f_out:
                    f_out.write(z.read(file_name))

            if os.path.exists(out_target):
                print(f"[{idx}/{total}] Already translated: {base_bname}")
                continue

            print(f"[{idx}/{total}] Processing: {base_bname}")
            t0 = time.time()
            try:
                process_single_image(raw_target, out_target)
                print(f"[✓] Completed in {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"[!] Error: {e}. Retrying with fallback...")
                time.sleep(2)
                try:
                    process_single_image(raw_target, out_target)
                    print(f"[✓] Completed on retry in {time.time()-t0:.2f}s")
                except Exception as e2:
                    print(f"[X] Page failed: {e2}. Keeping raw.")
                    import shutil
                    shutil.copyfile(raw_target, out_target)

    # Repack into CBZ
    print(f"[*] Repacking into {out_cbz}...")
    with zipfile.ZipFile(out_cbz, "w", zipfile.ZIP_DEFLATED) as z_out:
        for f in sorted(os.listdir(trans_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                z_out.write(os.path.join(trans_dir, f), arcname=f)
    print(f"[✓] CBZ created successfully at {out_cbz}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchidnaComicEngine Pro Autonomous CLI")
    parser.add_argument("--image", help="Single page image path")
    parser.add_argument("--out", help="Output path for single page")
    parser.add_argument("--cbz", help="Input CBZ comic file")
    parser.add_argument("--out-cbz", help="Output CBZ comic file")
    parser.add_argument("--workdir", default=r"C:\Users\ITcenter\ComicProjects\EchidnaEngine_Work")
    args = parser.parse_args()

    if args.image and args.out:
        process_single_image(args.image, args.out)
    elif args.cbz and args.out_cbz:
        translate_full_archive(args.cbz, args.out_cbz, args.workdir)
    else:
        parser.print_help()
