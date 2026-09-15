import random
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset, get_class_names


def test_dataset(name, split="test", n=10):
    print("=" * 80)
    print(f"Testing {name}")
    print("=" * 80)

    dataset = build_dataset(
        name,
        PROJECT_ROOT / "data",
        split=split,
        )

    class_names = get_class_names(dataset)

    indices = random.sample(range(len(dataset)), n)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]

        print(f"\nImage {i+1}")
        print(f"GT: {class_names[label]}")

        results = model.predict(
            img,
            verbose=False,
            conf=0.25,
        )

        names = results[0].names

        if len(results[0].boxes) == 0:
            print("YOLO: (no detections)")
            continue

        objs = []

        for box in results[0].boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())

            objs.append((names[cls], conf))

        # 去重，保留最高置信度
        best = {}

        for obj, conf in objs:
            if obj not in best or conf > best[obj]:
                best[obj] = conf

        best = sorted(
            best.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        print("YOLO:")

        for obj, conf in best:
            print(f"  {obj:15s} {conf:.3f}")


if __name__ == "__main__":

    model = YOLO("yolov8n.pt")

    test_dataset(
        "oxford_pet",
        split="test",
        n=10,
    )

    test_dataset(
        "ninco",
        split="test",
        n=10,
    )