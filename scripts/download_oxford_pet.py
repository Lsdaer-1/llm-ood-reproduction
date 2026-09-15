from pathlib import Path
from torchvision.datasets import OxfordIIITPet

ROOT = Path(__file__).resolve().parents[1] / "data"

OxfordIIITPet(root=str(ROOT), split="trainval", target_types="category", download=True)
OxfordIIITPet(root=str(ROOT), split="test", target_types="category", download=True)

print("Oxford-IIIT Pet downloaded to:", ROOT)