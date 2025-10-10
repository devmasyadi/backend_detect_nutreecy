# app/utils/crop_utils.py
import hashlib, uuid
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def file_hash(path: str) -> str:
    """Hash unik berdasarkan konten file."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]
    except Exception:
        return uuid.uuid4().hex[:8]

def _clip(v, lo, hi):
    return int(max(lo, min(hi, v)))

def save_crop_bbox(img_bgr, xyxy, padding, out_dir, hash_prefix, cls_name, conf, idx):
    """Simpan crop berdasarkan bounding box."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = _clip(x1 - padding, 0, w - 1)
    y1 = _clip(y1 - padding, 0, h - 1)
    x2 = _clip(x2 + padding, 0, w - 1)
    y2 = _clip(y2 + padding, 0, h - 1)
    crop = img_bgr[y1:y2, x1:x2].copy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe_cls = cls_name.replace(" ", "_")
    fp = Path(out_dir) / f"{hash_prefix}_{safe_cls}_{idx:03d}_{conf:.2f}.jpg"
    cv2.imwrite(str(fp), crop)
    return str(fp)

def save_crop_mask(img_bgr, mask, xyxy, padding, out_dir, hash_prefix, cls_name, conf, idx):
    """Simpan crop dengan mask (RGBA transparan)."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = _clip(x1 - padding, 0, w - 1)
    y1 = _clip(y1 - padding, 0, h - 1)
    x2 = _clip(x2 + padding, 0, w - 1)
    y2 = _clip(y2 + padding, 0, h - 1)
    crop_img = img_bgr[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask_crop
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe_cls = cls_name.replace(" ", "_")
    fp = Path(out_dir) / f"{hash_prefix}_{safe_cls}_{idx:03d}_{conf:.2f}.png"
    Image.fromarray(rgba).save(str(fp))
    return str(fp)
