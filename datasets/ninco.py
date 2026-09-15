from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


NINCO_ANIMAL_CLASSES = {
    "Caracal caracal caracal",
    "amphiuma_means",
    "araneus_gemma",
    "arctocephalus_galapagoensis",
    "batrachoseps_attenuatus",
    "ctenolepisma_longicaudata",
    "dendrolagus_lumholtzi",
    "haemulon_sciurus",
    "hippopus_hippopus",
    "lasionycteris_noctivagans",
    "lepomis_auritus",
    "leptoglossus_phyllopus",
    "octopus_bimaculoides",
    "octopus_rubescens",
    "ozotoceros_bezoarticus",
    "platycephalus_fuscus",
    "polistes_dominula",
    "pseudorca_crassidens",
    "sarpa_salpa",
    "sepia_apama",
    "sepia_officinalis",
    "sepioteuthis_australis",
    "skipper_caterpillar",
    "tapirus_bairdii",
    "triturus_marmoratus",
    "tursiops_aduncus",
}
HARD_NINCO_CLASSES = {
    "Caracal caracal caracal",
    "dendrolagus_lumholtzi",
    "sarpa_salpa",
    "ozotoceros_bezoarticus",
    "tapirus_bairdii",
}


class NINCO(Dataset):
    def __init__(self, root, split="test", subset="all"):
        root = Path(root)
        self.image_root = root / "NINCO" / "NINCO_OOD_classes"

        if not self.image_root.exists():
            raise FileNotFoundError(self.image_root)

        folders = sorted([
            p for p in self.image_root.iterdir()
            if p.is_dir()
        ])

        if subset == "animal":
            folders = [p for p in folders if p.name in NINCO_ANIMAL_CLASSES]
        elif subset != "all":
            raise ValueError(f"Unknown NINCO subset: {subset}")

        self.class_names = [p.name for p in folders]
        self.samples = []

        for label, folder in enumerate(folders):
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return img, label