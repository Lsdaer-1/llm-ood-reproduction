import random
import sys
from pathlib import Path

from ultralytics import YOLOWorld
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

vocabulary = [

    # ===== Oxford Pet =====

    "dog",
    "cat",
    "beagle",
    "boxer",
    "pug",
    "chihuahua",
    "pomeranian",
    "samoyed",
    "shiba inu",
    "yorkshire terrier",
    "scottish terrier",
    "english setter",
    "english cocker spaniel",
    "german shorthaired pointer",
    "great pyrenees",
    "american bulldog",
    "american pit bull terrier",
    "staffordshire bull terrier",
    "miniature pinscher",
    "newfoundland",
    "leonberger",
    "havanese",
    "keeshond",
    "japanese chin",

    "abyssinian cat",
    "bengal cat",
    "birman cat",
    "bombay cat",
    "british shorthair",
    "egyptian mau",
    "maine coon",
    "persian cat",
    "ragdoll",
    "russian blue",
    "siamese cat",
    "sphynx cat",

    # ===== NINCO =====

    "caracal",
    "bat",
    "dolphin",
    "killer whale",
    "sea lion",
    "tapir",
    "deer",
    "octopus",
    "cuttlefish",
    "squid",
    "fish",
    "sunfish",
    "insect",
    "spider",
    "caterpillar",
    "salamander",
    "bee",

    "bagpipe",
    "gramophone",
    "stapler",
    "cable",
    "walker",
    "high heels",

    "cupcake",
    "donut",
    "french fries",
    "spaghetti",
    "quesadilla",
    "waffle",
    "glass of milk",

    "fireworks",
    "pyramid",
    "hindu temple",
    "forest path",
    "road",
    "door",

    "flower",
    "pitcher plant",
    "cactus",
    "berry",
    "plant",
    "tree",

    # ===== COCO common =====

    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "table",

    "book",
    "backpack",
    "handbag",
    "suitcase",

    "cup",
    "bottle",
    "bowl",
    "wine glass",

    "fork",
    "knife",
    "spoon",

    "banana",
    "apple",
    "orange",
    "pizza",
    "sandwich",
    "cake",
    "hot dog",

    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",

    "traffic light",
    "fire hydrant",

    "clock",
    "tv",
    "laptop",
    "keyboard",
    "mouse",
    "cell phone",

    "toilet",
    "sink",

    "umbrella",

    "bird",

    "horse",
    "cow",
    "sheep",
    "bear",
    "zebra",
    "giraffe",

    "boat",

    "scissors",
    "toothbrush",

]
from utils.dataset_builder import build_dataset, get_class_names


def run_test(model, dataset_name, split="test", n=20, conf=0.15):
    print("=" * 80)
    print(f"Testing {dataset_name}, split={split}, conf={conf}")
    print("=" * 80)

    dataset = build_dataset(dataset_name, PROJECT_ROOT / "data", split=split)
    class_names = get_class_names(dataset)

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]

        results = model.predict(
            img,
            conf=conf,
            verbose=False,
        )

        print(f"\nImage {i+1}")
        print(f"GT: {class_names[label]}")

        boxes = results[0].boxes
        names = results[0].names

        if boxes is None or len(boxes) == 0:
            print("YOLO: (no detections)")
            continue

        best = {}

        for box in boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            obj = names[cls_id]

            if obj not in best or score > best[obj]:
                best[obj] = score

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)

        print("YOLO:")
        for obj, score in ranked[:5]:
            print(f"  {obj:15s} {score:.3f}")


if __name__ == "__main__":
    random.seed(0)

    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(vocabulary)


    run_test(model, "oxford_pet", split="test", n=20, conf=0.3)
    run_test(model, "ninco", split="hard", n=20, conf=0.3)