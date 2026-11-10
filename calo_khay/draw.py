"""Vẽ kết quả lên ảnh: viền màu từng món + nhãn tiếng Việt (gram, kcal) + tổng ở góc trên."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [(200, 240, 106), (255, 196, 0), (91, 160, 255), (190, 140, 255), (80, 220, 140),
           (255, 120, 170), (90, 220, 230), (255, 160, 90), (170, 200, 120), (240, 240, 240)]
FLAG = (255, 140, 40)


def _font(size):
    try:
        from matplotlib import font_manager  # ultralytics đã kéo theo matplotlib; DejaVu Sans có dấu tiếng Việt
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def draw(bgr: np.ndarray, res: dict) -> np.ndarray:
    H, W = bgr.shape[:2]
    over = bgr.copy()
    th = max(2, W // 300)
    for i, it in enumerate(res["items"]):
        col = FLAG if it["needs_confirm"] else PALETTE[i % len(PALETTE)]
        bgr_col = col[::-1]
        for p in it["polygons"]:
            if len(p) >= 3:
                pts = np.array(p, np.int32)
                cv2.fillPoly(over, [pts], bgr_col)
    img = cv2.addWeighted(over, 0.28, bgr, 0.72, 0)
    for i, it in enumerate(res["items"]):
        col = FLAG if it["needs_confirm"] else PALETTE[i % len(PALETTE)]
        for p in it["polygons"]:
            if len(p) >= 3:
                cv2.polylines(img, [np.array(p, np.int32)], True, col[::-1], th, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    fs = max(14, W // 55); f = _font(fs); fb = _font(int(fs * 1.4))
    for i, it in enumerate(res["items"]):
        col = FLAG if it["needs_confirm"] else PALETTE[i % len(PALETTE)]
        big = max(it["polygons"], key=len)
        if len(big) < 3:
            continue
        x, y = np.array(big).mean(0).astype(int)
        txt = (f"? {it['name']}" if it["needs_confirm"] else it["name"]) + f"  {it['grams']}g · {it['kcal']}kcal"
        l, t, r, b = dr.textbbox((x, y), txt, font=f, anchor="mm")
        dr.rounded_rectangle((l - 6, t - 4, r + 6, b + 4), radius=fs // 2, fill=(15, 18, 16), outline=col, width=2)
        dr.text((x, y), txt, font=f, fill=(255, 255, 255), anchor="mm")
    tot = res["total"]
    head = f"{tot['kcal']} kcal  ·  {tot['grams']} g  (±{tot['error_pct']}%/món)"
    l, t, r, b = dr.textbbox((16, 16), head, font=fb)
    dr.rectangle((0, 0, r + 16, b + 14), fill=(11, 15, 13))
    dr.text((16, 16), head, font=fb, fill=(200, 240, 106))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
