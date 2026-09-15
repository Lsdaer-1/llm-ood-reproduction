import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset, get_class_names

client = OpenAI()


def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


import time

def detect_objects(img, model_name):

    image = pil_to_base64(img)

    for attempt in range(3):

        try:

            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": """You are an object detector.

Identify only clearly visible physical objects in this image.

Rules:
- Output ONLY object names.
- One object per line.
- Maximum 5 objects.
- Use lowercase.
- Use singular nouns.
- No explanations.
- No punctuation.
- Do not output adjectives.
- If nothing is clearly visible, output unknown."""
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{image}",
                            },
                        ],
                    }
                ],
            )

            objects = []

            for line in response.output_text.splitlines():
                line = line.strip().lower()
                line = line.replace("-", "").replace(".", "").replace(",", "")

                if line:
                    objects.append(line)

            return objects[:5]

        except Exception as e:

            if attempt == 2:
                raise e

            print(f"Retry {attempt+1}/3 ...")
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset = build_dataset(args.dataset, PROJECT_ROOT / "data", split=args.split)
    class_names = get_class_names(dataset)

    if args.output is None:
        out = PROJECT_ROOT / "results" / f"gpt_objects_{args.dataset}_{args.split}.json"
    else:
        out = Path(args.output)

    out.parent.mkdir(exist_ok=True)

    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
    else:
        result = {}

    n = min(args.max_samples, len(dataset))

    for idx in tqdm(range(n)):
        key = str(idx)

        if key in result:
            if (
                 isinstance(result[key], dict)
                 and "error" not in result[key]
                 and len(result[key].get("objects", [])) > 0
            ):
                continue

            print(f"Retry image {key}")

        img, label = dataset[idx]

        try:
            objs = detect_objects(img, args.model)
            result[key] = {
                "label": int(label),
                "class_name": class_names[label],
                "objects": objs,
            }

            if idx < 10:
                print(idx, class_names[label], "=>", objs)

        except Exception as e:
            print(f"Error on image {idx}: {e}")
            result[key] = {
                "label": int(label),
                "class_name": class_names[label],
                "objects": [],
                "error": str(e),
            }

        if idx % 10 == 0:
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()