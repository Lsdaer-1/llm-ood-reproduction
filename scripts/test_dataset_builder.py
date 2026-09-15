import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset, get_class_names


data_root = PROJECT_ROOT / "data"

id_dataset = build_dataset("tiny_imagenet", data_root, split="val")
ood_dataset = build_dataset("dtd", data_root)

print("ID dataset:", len(id_dataset))
print("ID classes:", len(get_class_names(id_dataset)))
print("ID first classes:", get_class_names(id_dataset)[:5])

print("OOD dataset:", len(ood_dataset))
print("OOD classes:", len(get_class_names(ood_dataset)))
print("OOD first classes:", get_class_names(ood_dataset)[:5])

img, label = id_dataset[0]
print("ID image size:", img.size)
print("ID label:", label)
print("ID class:", get_class_names(id_dataset)[label])

img, label = ood_dataset[0]
print("OOD image size:", img.size)
print("OOD label:", label)
print("OOD class:", get_class_names(ood_dataset)[label])