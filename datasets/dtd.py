from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class DTD(Dataset):
    def __init__(self, root):
        self.root = Path(root) / "dtd"
        self.image_root = self.root / "images"

        self.class_names = sorted([
            p.name for p in self.image_root.iterdir()
            if p.is_dir()
        ])

        self.class_to_label = {
            name: i for i, name in enumerate(self.class_names)
        }

        self.samples = []

        for class_name in self.class_names:
            img_dir = self.image_root / class_name
            label = self.class_to_label[class_name]

            for img_path in sorted(img_dir.glob("*.jpg")):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return img, label