import os

from openai import OpenAI


def main() -> None:
    api_key = os.getenv("OPEN_MODEL_KEY")
    if not api_key:
        raise RuntimeError("Set OPEN_MODEL_KEY before running this sample.")

    client = OpenAI(
        base_url="https://api.openmodel.ai/v1",
        api_key=api_key,
    )

    response = client.responses.create(
        model="deepseek-v4-flash",
        input="Hello, who are you?",
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
