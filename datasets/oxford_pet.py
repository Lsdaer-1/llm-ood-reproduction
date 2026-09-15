from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class OxfordPet(Dataset):
    def __init__(self, root, split="test"):
        self.root = Path(root) / "oxford-iiit-pet"

        self.image_root = self.root / "images"
        self.ann_root = self.root / "annotations"

        if split == "train":
            ann_file = self.ann_root / "trainval.txt"
        elif split == "test":
            ann_file = self.ann_root / "test.txt"
        else:
            raise ValueError(split)

        self.samples = []

        class_names = {}

        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split()

                image_name = parts[0]
                label = int(parts[1]) - 1

                img_path = self.image_root / f"{image_name}.jpg"

                self.samples.append((img_path, label))

                # 去掉最后的编号
                cls = "_".join(image_name.split("_")[:-1])
                cls = cls.replace("_", " ").lower()

                class_names[label] = cls

        self.class_names = [
            class_names[i]
            for i in range(len(class_names))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        return img, label