import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.oxford_pet import OxfordPet

dataset = OxfordPet(
    root=PROJECT_ROOT / "data",
    split="test",
)

print("num samples:", len(dataset))
print("num classes:", len(dataset.class_names))
print(dataset.class_names[:10])

img, label = dataset[0]

print(img.size)
print(label)
print(dataset.class_names[label])
