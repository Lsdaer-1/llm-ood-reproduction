import base64
from openai import OpenAI

client = OpenAI()


def detect_objects(image_path):

    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode()

    response = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text":
                        """List every visible physical object in this image.
                        
                        Rules:
                        
                        - Output ONLY object names.
                        - One object per line.
                        - Use singular nouns.
                        - Do NOT explain.
                        - Do NOT infer invisible objects.
                        - Do NOT output adjectives.
                        - Do NOT output complete sentences."""
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image}"
                }
            ]
        }]
    )

    text = response.output_text.strip()

    objects = [
        x.strip().lower()
        for x in text.splitlines()
        if x.strip()
    ]

    return objects


if __name__ == "__main__":

    objs = detect_objects("dog.png")

    print(objs)