import base64
import json
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from torchvision.datasets import CIFAR10
from tqdm import tqdm

client = OpenAI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT = PROJECT_ROOT / "results" / "gpt_objects_cifar.json"
OUT.parent.mkdir(exist_ok=True)


def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def detect_objects(img):

    image = pil_to_base64(img)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """
Identify all visible physical objects.

Rules:

- Output ONLY object names.
- One object per line.
- Maximum 5 objects.
- Use lowercase.
- Use singular nouns.
- No explanation.
- No punctuation.
- If uncertain, do not include it.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image}"
                    }
                ]
            }
        ]
    )

    objects = []

    for line in response.output_text.splitlines():

        line = line.strip().lower()

        if not line:
            continue

        line = line.replace("-", "")
        line = line.replace(".", "")

        objects.append(line)

    return objects[:5]


def main():

    dataset = CIFAR10(
        root=str(PROJECT_ROOT / "scripts"/"data"),
        train=False,
        download=False,
        transform=None,
    )

    if OUT.exists():
        result = json.loads(OUT.read_text())
    else:
        result = {}

    for idx in tqdm(range(20)):

        key = str(idx)

        if key in result:
            continue

        img, label = dataset[idx]

        try:

            objs = detect_objects(img)

            result[key] = objs

            if idx < 10:
                print(idx, objs)

        except Exception as e:

            print(f"Error on image {idx}: {e}")

            result[key] = []

        if idx % 20 == 0:

            OUT.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()