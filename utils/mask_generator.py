import cv2
import numpy as np
import os

def generate_masks(src_dir="data/annotated_images", dst_dir="data/masks"):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red highlight thresholds — adjust if your color is different
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        out_path = os.path.join(dst_dir, os.path.splitext(fname)[0] + "_mask.png")
        cv2.imwrite(out_path, mask)
        print(f"Mask created for {fname}")

if __name__ == "__main__":
    generate_masks()
