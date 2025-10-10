# main.py
import argparse, sys, subprocess
import tempfile, shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.detect_yolo import detect_and_analyze

def create_app(config):
    """
    Inisialisasi FastAPI app dengan konfigurasi dari argparse.
    """
    api_prefix = config.api_prefix  # contoh: "/api/v1"
    base_url = config.base_url.rstrip("/")  # contoh: "https://app.masyadi.com"
    crop_dir = Path(config.crop_dir)
    static_prefix = config.static_prefix.strip("/")
    if not static_prefix:
        static_prefix = crop_dir.name  # default ke nama folder crop_dir

    app = FastAPI(
        title="Nutreecy Food Detection API",
        description="Deteksi makanan dan analisis gizi berbasis YOLO"
    )

    # 1) Mount static untuk melayani file di crop_dir
    #    Akses: {base_url}/{static_prefix}/<relative_path_dari_crop_dir>
    crop_dir.mkdir(parents=True, exist_ok=True)
    app.mount(f"/{static_prefix}", StaticFiles(directory=str(crop_dir), html=False), name="crops")

    @app.post(f"{api_prefix}/detect")
    async def detect_food(file: UploadFile = File(...)):
        """Endpoint deteksi makanan."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Jalankan YOLO
        result = detect_and_analyze(
            model_path=config.model,
            source=tmp_path,
            conf=config.conf,
            imgsz=config.imgsz,
            iou=config.iou,
            device=config.device,
            save_crops=config.save_crops,
            crop_dir=str(config.crop_dir),
            crop_padding=config.crop_padding,
            alpha_mask=config.alpha_mask,
            base_url=config.base_url,
            static_prefix=config.static_prefix
        )

        return result

    @app.get(f"{api_prefix}/ping")
    async def ping():
        """Cek status API."""
        return {
            "status": "ok",
            "model": config.model,
            "device": config.device,
            "api_prefix": api_prefix or "/",
            "port": config.port,
            "base_url": base_url,
            "static_prefix": static_prefix,
            "static_mounted": f"/{static_prefix}",
        }

    return app


def parse_args():
    """
    Argumen CLI — konfigurasi API, static files, dan YOLO inference.
    """
    p = argparse.ArgumentParser(description="Nutreecy Food Detection API Server")

    # ---- API args ----
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host binding (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8003, help="Port server (default: 8003)")
    p.add_argument("--base-url", type=str, default="http://localhost:8003",
                   help="Base domain untuk URL absolut (contoh: https://app.masyadi.com)")
    p.add_argument("--api-prefix", type=str, default="", help="Prefix endpoint API (contoh: /api/v1)")
    p.add_argument("--reload", action="store_true", help="Aktifkan reload otomatis (mode dev)")

    # ---- Static files ----
    p.add_argument("--static-prefix", type=str, default="", help="URL prefix untuk static crops (default: nama folder crop_dir)")

    # ---- YOLO config ----
    p.add_argument("--model", default="model/food_detection.pt", help="Path atau nama model YOLO")
    p.add_argument("--src", default=None, help="Sumber input (tidak digunakan untuk FastAPI upload)")
    p.add_argument("--imgsz", type=int, default=640, help="Ukuran gambar input YOLO")
    p.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    p.add_argument("--device", default="cpu", help="'cpu' atau '0' untuk GPU")
    p.add_argument("--save-crops", action="store_true", help="Simpan hasil crop objek")
    p.add_argument("--crop-dir", default="output", help="Folder untuk hasil crop")
    p.add_argument("--crop-padding", type=int, default=4, help="Padding di sekitar bounding box")
    p.add_argument("--alpha-mask", action="store_true", help="Aktifkan RGBA mask (model seg)")

    return p.parse_args()


if __name__ == "__main__":
    config = parse_args()

    if config.reload:
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--reload", "--host", config.host, "--port", str(config.port)
        ]
        print("🔁 Reload mode aktif:", " ".join(cmd))
        subprocess.run(cmd)
        sys.exit(0)

    app = create_app(config)
    print(f"📦 Model: {config.model} | Device: {config.device} | Conf: {config.conf}")
    uvicorn.run(app, host=config.host, port=config.port, reload=False)
