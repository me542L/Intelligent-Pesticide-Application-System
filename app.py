from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from utils.metrics import infection_percentage, spray_level_from_pct
from pyngrok import ngrok

# Load trained model
MODEL_PATH = "runs/segment/train/weights/best.pt"
model = YOLO(MODEL_PATH)

app = Flask(__name__)

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
        return jsonify({"infection_percent": float(pct), "recommendation": level})

    return jsonify({"infection_percent": 0.0, "recommendation": "Healthy"})

if __name__ == "__main__":
    # 🔗 Create a public tunnel so you can access it from your browser
    public_url = ngrok.connect(5000).public_url
    print("🌐 Public URL:", public_url)
    app.run(port=5000)


  
