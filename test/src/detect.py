from app.detect_yolo import detect_and_analyze
import json

if __name__ == "__main__":
    data = detect_and_analyze(
        model_path="model/food_detection.pt",
        source="test/image/0.jpg",
        conf=0.7,
        save_crops=True
    )
    print(json.dumps(data, indent=2, ensure_ascii=False))
