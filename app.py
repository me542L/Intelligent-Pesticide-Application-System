from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from utils.metrics import infection_percentage, spray_level_from_pct

app = Flask(__name__)
model = YOLO("runs/segment/train/weights/best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    results = model(img)[0]

    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        mask = (masks[0] > 0.5).astype(np.uint8)
        bbox = results.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox
        pct = infection_percentage(mask, bbox=(x1, y1, x2 - x1, y2 - y1))
        level = spray_level_from_pct(pct)
        return jsonify({"infection_percent": pct, "recommendation": level})

    return jsonify({"infection_percent": 0, "recommendation": "Healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
