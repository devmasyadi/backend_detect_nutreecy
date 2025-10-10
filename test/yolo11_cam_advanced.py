import time
import cv2
import argparse
from ultralytics import YOLO
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/my_model_2.pt", help="Path/nama model YOLOv11")
    p.add_argument("--src", type=int, default=0, help="Index webcam (0/1/2)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.8, help="Tampilkan hanya hasil di atas threshold ini")
    p.add_argument("--save", action="store_true", help="Simpan hasil ke file mp4")
    p.add_argument("--out", default="out.mp4", help="Nama file output")
    p.add_argument("--cpu", action="store_true", help="Paksa pakai CPU")
    return p.parse_args()

def main():
    args = parse_args()

    # Pilih device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else 0
    model = YOLO(args.model)
    model.to(device)

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera index {args.src}")

    # Set resolusi (opsional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference
        results = model(frame, stream=True, imgsz=args.imgsz, conf=args.conf)

        for r in results:
            # Filter hasil hanya di atas threshold
            boxes = r.boxes
            mask = boxes.conf > args.threshold
            filtered_boxes = boxes[mask]

            # Gambar ulang manual hanya untuk box lolos threshold
            annotated = frame.copy()
            for box in filtered_boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id] if cls_id in model.names else str(cls_id)

                x1, y1, x2, y2 = xyxy
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Hitung FPS
            now = time.time()
            dt = now - prev
            prev = now
            fps = 1.0 / dt if dt > 0 else fps

            txt = f"FPS: {fps:.1f} | Model: {args.model} | Device: {'CPU' if device=='cpu' else 'CUDA'} | Thresh: {args.threshold}"
            cv2.putText(annotated, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("YOLOv11 - Webcam (Filtered)", annotated)

            if writer:
                writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
