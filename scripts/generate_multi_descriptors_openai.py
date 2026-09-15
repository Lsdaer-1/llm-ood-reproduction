import json
from pathlib import Path
from openai import OpenAI

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

client = OpenAI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "descriptors" / "cifar10_gpt4.1_multi_descriptors.json"
OUT.parent.mkdir(exist_ok=True)


def generate_for_class(class_name: str) -> list[str]:
    prompt = f"""Q: What are useful visual features for distinguishing a lemur in a photo?

A: There are several useful visual features to tell there is a lemur in a photo:
- furry bodies
- long tail
- large eyes

Q: What are useful visual features for distinguishing a {class_name} in a photo?

A: There are several useful visual features to tell there is a {class_name} in a photo:
-"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        temperature=0.7,
        input=prompt,
        max_output_tokens=100,
    )

    print("=" * 60)
    print(class_name)
    print(response.output_text)
    print("=" * 60)

    text = response.output_text
    lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("there are several useful visual features"):
            continue
        if line.startswith("-"):
            lines.append(line.lstrip("-").strip())
        elif len(lines) < 10:
            lines.append(line.strip("0123456789. ").strip())

    return lines[:10]


def main():
    result = {}

    for c in CLASSES:
        result[c] = []

        for i in range(10):
            print(f"Generating descriptors for {c}, round {i + 1}...")
            result[c].append(generate_for_class(c))

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()