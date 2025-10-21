from ultralytics import YOLO

def train_yolo_seg():
    model = YOLO("yolov8n-seg.pt")  # lightweight segmentation model
    model.train(data="data/data.yaml", epochs=100, imgsz=640, batch=8)

if __name__ == "__main__":
    train_yolo_seg()
