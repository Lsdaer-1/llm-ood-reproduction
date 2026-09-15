from ultralytics import YOLO
from torchvision.datasets import CIFAR10
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

model = YOLO("yolov8n.pt")

dataset = CIFAR10(
    root=str(PROJECT_ROOT / "scripts"/"data"),
    train=False,
    download=False,
    transform=None,
)

img, label = dataset[0]

results = model.predict(
    source=img,
    verbose=True,
)

for r in results:
    names = r.names
    boxes = r.boxes

    print("Detected objects:")
    for cls_id, conf in zip(boxes.cls, boxes.conf):
        print(names[int(cls_id)], float(conf))