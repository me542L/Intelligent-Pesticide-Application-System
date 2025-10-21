import cv2
import numpy as np
from ultralytics import YOLO
from utils.metrics import infection_percentage, spray_level_from_pct

model = YOLO("runs/segment/train/weights/best.pt")  # adjust path after training

def analyze_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    display = img.copy()
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        for i, m in enumerate(masks):
            mask = (m > 0.5).astype(np.uint8)
            bbox = results.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            pct = infection_percentage(mask, bbox=(x1, y1, x2-x1, y2-y1))
            level = spray_level_from_pct(pct)

            label = f"{level} ({pct:.1f}%)"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            overlay = np.zeros_like(display)
            overlay[:, :, 2] = mask * 255
            display = cv2.addWeighted(display, 1.0, overlay, 0.4, 0)

    cv2.imshow("Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_image("data/test_leaf.jpg")
