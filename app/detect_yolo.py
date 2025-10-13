# app/detect_yolo.py
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from app.utils.crop_utils import file_hash, save_crop_bbox, save_crop_mask
from app.utils.nutrition_db import get_nutrition, NUTRITION_DB

def detect_and_analyze(
    model_path,
    source,
    conf=0.7,
    imgsz=640,
    iou=0.45,
    device="cpu",
    save_crops=True,                 # <- default True biar crop_path tidak null
    crop_dir="output",
    crop_padding=4,
    alpha_mask=False,
    base_url: str | None = None,     # <- domain produksi (https://app.masyadi.com)
    static_prefix: str | None = None # <- prefix URL untuk static (mis. "output" atau "media")
):
    """
    Jalankan deteksi YOLO, filter hanya kelas yang ada di NUTRITION_DB,
    dedup per class (ambil confidence tertinggi), dan kembalikan JSON.

    Jika base_url diisi, crop_path akan dikonversi menjadi URL absolut:
        {base_url}/{static_prefix}/{relative_path_dari_crop_dir}
    """
    model = YOLO(model_path)
    names = model.model.names if hasattr(model.model, "names") else model.names

    # siapkan prefix untuk URL statik
    if static_prefix is None or not str(static_prefix).strip("/"):
        static_prefix = Path(crop_dir).name
    static_prefix = str(static_prefix).strip("/")

    results = model.predict(
        source=source, imgsz=imgsz, conf=conf, iou=iou,
        device=device, save=False, verbose=False
    )

    all_images = []

    for r in results:
        img_name = Path(r.path).name if r.path else "image"
        img_hash = file_hash(r.path)
        img_bgr = r.orig_img if hasattr(r, "orig_img") else None

        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            full_masks = None
            if alpha_mask and getattr(r, "masks", None) is not None and r.masks is not None:
                mdata = r.masks.data.cpu().numpy()
                H, W = r.masks.orig_shape
                full_masks = [(cv2.resize(m, (W, H)) > 0.5).astype(np.uint8) for m in mdata]

            for i, box in enumerate(xyxy):
                cname = (names[int(clss[i])].strip().lower()
                         if not isinstance(names, dict)
                         else str(names.get(int(clss[i]), int(clss[i]))).strip().lower())

                # hanya yang ada gizinya
                if cname not in NUTRITION_DB:
                    continue

                conf_i = float(confs[i])
                bbox = [float(x) for x in box.tolist()]
                nutri = get_nutrition(cname)

                crop_path_local = None
                if save_crops and img_bgr is not None:
                    if full_masks is not None and i < len(full_masks):
                        crop_path_local = save_crop_mask(
                            img_bgr, full_masks[i], box, crop_padding, str(crop_dir),
                            img_hash, cname, conf_i, i
                        )
                    else:
                        crop_path_local = save_crop_bbox(
                            img_bgr, box, crop_padding, str(crop_dir),
                            img_hash, cname, conf_i, i
                        )

                # konversi ke URL absolut jika base_url diberikan
                crop_path_final = crop_path_local
                if crop_path_local and base_url:
                    try:
                        rel = Path(crop_path_local).resolve().relative_to(crop_dir.resolve()).as_posix()
                    except Exception:
                        rel = Path(crop_path_local).name
                    crop_path_final = f"{base_url.rstrip('/')}/{static_prefix}/{rel}"

                detections.append({
                    "class_name": cname,
                    "confidence": conf_i,
                    "bbox_xyxy": bbox,
                    "crop_path": crop_path_final,
                    "nutrition": {
                        "portion": nutri["portion"],
                        "calories": nutri["calories"],
                        "protein": nutri["protein"],
                        "carbs":   nutri["carbs"],
                        "fat":     nutri["fat"],
                    }
                })

        # dedup per class (confidence tertinggi)
        best = {}
        for d in detections:
            c = d["class_name"]
            if c not in best or d["confidence"] > best[c]["confidence"]:
                best[c] = d
        filtered = list(best.values())

        if filtered:
            all_images.append({"image": img_name, "hash": img_hash, "detections": filtered})

    return {"results": all_images}
