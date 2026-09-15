import argparse
import json
import sys
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset, get_class_names

client = OpenAI()


def generate_for_class(class_name: str, model_name: str) -> list[str]:
    prompt = f"""

You are describing the Oxford-IIIT Pet dataset.

The class name is:

{class_name}

This class ALWAYS refers to the Oxford-IIIT Pet breed.

Do NOT interpret it as any other animal or object.

For example:

- Bengal means Bengal cat.

- Boxer means Boxer dog.

- Chihuahua means Chihuahua dog.

List EXACTLY 6 visual descriptors that are useful for recognizing this class in an image.

Requirements:

- Output ONLY the descriptors.

- One descriptor per line.

- No numbering.

- No bullets.

- No explanations.

- No introductory sentence.

- No "Answer:".

- No "There are several useful visual features..."

- No assumptions.

- Do not mention multiple possible meanings.

- Every descriptor should describe appearance only.

Example output:

short smooth coat

large upright ears

black nose

muscular body

drooping ears

white chest

"""

    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0.7,
        max_output_tokens=100,
    )

    lines = []
    for line in response.output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("there are several useful visual features"):
            continue
        if line.startswith("-"):
            lines.append(line.lstrip("-").strip())
        else:
            lines.append(line.strip("0123456789. ").strip())

    return lines[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oxford_pet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_sets", type=int, default=10)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset = build_dataset(args.dataset, PROJECT_ROOT / "data", split=args.split)
    class_names = get_class_names(dataset)

    if args.output is None:
        out = PROJECT_ROOT / "descriptors" / f"{args.dataset}_gpt_descriptors.json"
    else:
        out = Path(args.output)

    out.parent.mkdir(exist_ok=True)

    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
    else:
        result = {}

    for cls in class_names:
        if cls not in result:
            result[cls] = []

        while len(result[cls]) < args.num_sets:
            i = len(result[cls]) + 1
            print(f"Generating {cls}, set {i}/{args.num_sets}")
            desc = generate_for_class(cls, args.model)
            print(desc)
            result[cls].append(desc)

            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved to {out}")


if __name__ == "__main__":
    main()