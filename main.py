import os

import anthropic


def main() -> None:
    api_key = os.getenv("OPEN_MODEL_KEY")
    if not api_key:
        raise RuntimeError("Set OPEN_MODEL_KEY before running this sample.")

    client = anthropic.Anthropic(
        base_url="https://api.openmodel.ai",
        api_key=api_key,
    )

    response = client.messages.create(
        model="deepseek-v4-flash",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, who are you?"}],
    )
    print(response.content[0].text)


if __name__ == "__main__":
    main()
