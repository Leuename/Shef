import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA


DEFAULT_MODEL = "deepseek-ai/deepseek-v4-pro"


def main() -> None:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY before running this sample.")

    model = ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        temperature=0.35,
        max_completion_tokens=1024,
        model_kwargs={"chat_template_kwargs": {"thinking": False}},
    )

    response = model.invoke("Hello, who are you?")
    print(response.content)


if __name__ == "__main__":
    main()
