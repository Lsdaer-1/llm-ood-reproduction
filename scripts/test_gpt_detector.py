import base64
from pathlib import Path

from openai import OpenAI
from torchvision.datasets import CIFAR10

client = OpenAI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

dataset = CIFAR10(
    root=str(PROJECT_ROOT /"scripts"/"data"),
    train=False,
    download=False,
    transform=None,
)

img, label = dataset[0]

tmp = PROJECT_ROOT / "tmp_test.png"
img.save(tmp)

with open(tmp, "rb") as f:
    image = base64.b64encode(f.read()).decode()

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": """Identify the visible objects.

Rules:
- Return ONLY object names.
- One object per line.
- Maximum 5 objects.
- No explanation.
- No sentences.
- Use singular nouns.
- If unsure, return your best guess."""
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image}"
                }
            ]
        }
    ]
)

print(response.output_text)