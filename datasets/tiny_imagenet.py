from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    def __init__(self, root, split="val"):
        self.root = Path(root) / "tiny-imagenet-200"
        self.split = split

        self.wnids = (self.root / "wnids.txt").read_text().splitlines()

        self.wnid_to_label = {wnid: i for i, wnid in enumerate(self.wnids)}

        self.wnid_to_name = {}
        for line in (self.root / "words.txt").read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                wnid = parts[0]
                name = parts[1].split(",")[0].strip()
                self.wnid_to_name[wnid] = name

        self.class_names = [
            self.wnid_to_name.get(wnid, wnid)
            for wnid in self.wnids
        ]

        self.samples = []

        if split == "train":
            self._load_train()
        elif split == "val":
            self._load_val()
        else:
            raise ValueError(f"Unknown split: {split}")

    def _load_train(self):
        train_dir = self.root / "train"

        for wnid in self.wnids:
            img_dir = train_dir / wnid / "images"
            label = self.wnid_to_label[wnid]

            for img_path in sorted(img_dir.glob("*.JPEG")):
                self.samples.append((img_path, label))

    def _load_val(self):
        val_dir = self.root / "val"
        ann_path = val_dir / "val_annotations.txt"

        for line in ann_path.read_text().splitlines():
            parts = line.split("\t")
            img_name = parts[0]
            wnid = parts[1]

            img_path = val_dir / "images" / img_name
            label = self.wnid_to_label[wnid]

            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        return img, label