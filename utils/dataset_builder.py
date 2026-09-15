from pathlib import Path

from datasets.tiny_imagenet import TinyImageNet
from datasets.dtd import DTD
from datasets.oxford_pet import OxfordPet
from datasets.ninco import NINCO
def build_dataset(name: str, root, split: str = "val"):
    root = Path(root)

    name = name.lower()
    if name == "oxford_pet":
        return OxfordPet(root=root, split=split)
    if name == "ninco":
        subset = "animal" if split == "animal" else "all"
        return NINCO(root=root, split=split, subset=subset)

    if name == "tiny_imagenet":
        return TinyImageNet(root=root, split=split)

    if name == "dtd":
        return DTD(root=root)

    raise ValueError(f"Unknown dataset: {name}")


def get_class_names(dataset):
    if hasattr(dataset, "class_names"):
        return dataset.class_names

    raise ValueError("Dataset does not have class_names attribute.")