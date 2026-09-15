import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dtd import DTD


dataset = DTD(root=PROJECT_ROOT / "data")

print("num samples:", len(dataset))
print("num classes:", len(dataset.class_names))
print("first 10 classes:")
print(dataset.class_names[:10])

img, label = dataset[0]
print("first image size:", img.size)
print("first label:", label)
print("first class name:", dataset.class_names[label])