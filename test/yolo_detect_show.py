# pip install ultralytics opencv-python pillow numpy torch
import argparse, hashlib, uuid, json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ====== Tabel gizi (per porsi) ======
NUTRITION_DB = {
    "ayam goreng":   {"portion": "1 potong (~100 g)", "calories": 260, "protein": 28,  "carbs": 8,   "fat": 15},
    "bakso":         {"portion": "5 butir (~150 g)",  "calories": 220, "protein": 18,  "carbs": 14,  "fat": 10},
    "bakwan":        {"portion": "1 pc (~60 g)",      "calories": 180, "protein": 3,   "carbs": 15,  "fat": 12},
    "bubur ayam":    {"portion": "1 mangkuk",         "calories": 320, "protein": 14,  "carbs": 45,  "fat": 9},
    "nasi":          {"portion": "1 porsi (150 g)",   "calories": 204, "protein": 4,   "carbs": 45,  "fat": 0.4},
    "sambal":        {"portion": "1 sdm (15 g)",      "calories": 15,  "protein": 0.3, "carbs": 3,   "fat": 0.2},
    "sate":          {"portion": "5 tusuk (~150 g)",  "calories": 300, "protein": 25,  "carbs": 10,  "fat": 15},
    "tahu goreng":   {"portion": "1 potong (80 g)",   "calories": 140, "protein": 10,  "carbs": 6,   "fat": 8},
    "telur rebus":   {"portion": "1 butir (50 g)",    "calories": 78,  "protein": 6.3, "carbs": 0.6, "fat": 5.3},
    "tempe goreng":  {"portion": "1 potong (50 g)",   "calories": 180, "protein": 10,  "carbs": 8,   "fat": 10},

    # === tambahan buah-buahan ===
    "jeruk":         {"portion": "1 buah sedang (~130 g)", "calories": 62,  "protein": 1.2, "carbs": 15.4, "fat": 0.2},
    "apel":          {"portion": "1 buah sedang (~182 g)", "calories": 95,  "protein": 0.5, "carbs": 25,   "fat": 0.3},
    "alpukat":       {"portion": "1 buah sedang (~150 g)", "calories": 240, "protein": 3,   "carbs": 12.8, "fat": 22},
}


def parse_args():
    p = argparse.ArgumentParser("YOLO → JSON (unique per class + nutrition filter)")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--src", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.7)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--crop-dir", default="runs/crops")
    p.add_argument("--crop-padding", type=int, default=4)
    p.add_argument("--alpha-mask", action="store_true")
    p.add_argument("--json-out", default=None)
    return p.parse_args()

def file_hash(path: str):
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]
    except Exception:
        return uuid.uuid4().hex[:8]

def _clip(v, lo, hi): return int(max(lo, min(hi, v)))

def _save_crop_bbox(img_bgr, xyxy, padding, out_dir, hash_prefix, cls_name, conf, idx):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = _clip(x1 - padding, 0, w-1)
    y1 = _clip(y1 - padding, 0, h-1)
    x2 = _clip(x2 + padding, 0, w-1)
    y2 = _clip(y2 + padding, 0, h-1)
    crop = img_bgr[y1:y2, x1:x2].copy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe_cls = cls_name.replace(" ", "_")
    fp = Path(out_dir)/f"{hash_prefix}_{safe_cls}_{idx:03d}_{conf:.2f}.jpg"
    cv2.imwrite(str(fp), crop)
    return str(fp)

def _save_crop_mask(img_bgr, mask, xyxy, padding, out_dir, hash_prefix, cls_name, conf, idx):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = _clip(x1 - padding, 0, w-1)
    y1 = _clip(y1 - padding, 0, h-1)
    x2 = _clip(x2 + padding, 0, w-1)
    y2 = _clip(y2 + padding, 0, h-1)
    crop_img = img_bgr[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask_crop
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe_cls = cls_name.replace(" ", "_")
    fp = Path(out_dir)/f"{hash_prefix}_{safe_cls}_{idx:03d}_{conf:.2f}.png"
    Image.fromarray(rgba).save(str(fp))
    return str(fp)

def nutrition_for(cls_name: str):
    key = cls_name.strip().lower()
    info = NUTRITION_DB.get(key)
    if not info: return None
    return {
        "portion": info["portion"],
        "calories": info["calories"],
        "protein": info["protein"],
        "carbs": info["carbs"],
        "fat": info["fat"],
    }

def main():
    a = parse_args()
    model = YOLO(a.model)
    names = model.model.names if hasattr(model.model, "names") else model.names

    results = model.predict(
        source=a.src, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
        device=a.device, save=False, verbose=False
    )

    all_images = []
    for r in results:
        img_name = Path(r.path).name if r.path else "image"
        img_hash = file_hash(r.path)
        img_bgr = r.orig_img if hasattr(r, "orig_img") else None

        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)

            full_masks = None
            if a.alpha_mask and getattr(r, "masks", None) is not None and r.masks is not None:
                mdata = r.masks.data.cpu().numpy()
                H, W = r.masks.orig_shape
                full_masks = [(cv2.resize(m, (W, H)) > 0.5).astype(np.uint8) for m in mdata]

            for i, box in enumerate(xyxy):
                cid = int(cls[i])
                cname = names.get(cid) if isinstance(names, dict) else names[cid]
                cname = cname.strip().lower()
                if cname not in NUTRITION_DB:
                    continue  # skip yang tidak ada gizi
                conf_i = float(conf[i])
                bbox = [float(x) for x in box.tolist()]
                nutri = nutrition_for(cname)
                crop_path = None
                if a.save_crops and img_bgr is not None:
                    if full_masks is not None and i < len(full_masks):
                        crop_path = _save_crop_mask(img_bgr, full_masks[i], box, a.crop_padding, a.crop_dir, img_hash, cname, conf_i, i)
                    else:
                        crop_path = _save_crop_bbox(img_bgr, box, a.crop_padding, a.crop_dir, img_hash, cname, conf_i, i)
                detections.append({
                    "class_name": cname,
                    "confidence": conf_i,
                    "bbox_xyxy": bbox,
                    "crop_path": crop_path,
                    "nutrition": nutri
                })

        # deduplicate per class: simpan yang conf tertinggi
        best = {}
        for d in detections:
            c = d["class_name"]
            if c not in best or d["confidence"] > best[c]["confidence"]:
                best[c] = d
        filtered = list(best.values())
        if filtered:
            all_images.append({"image": img_name, "hash": img_hash, "detections": filtered})

    payload = {"results": all_images}
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON saved to {a.json_out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
