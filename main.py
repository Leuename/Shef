# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model="deepseek-v4-flash",
#     api_key="om-WDypCJiTysfsM9ZvZeweNfB1Wob6d17YnZDy4BhTH",
#     base_url="https://api.openmodel.ai/v1",
#     temperature=0.7,
# )

# response = llm.invoke("Explain Python decorators in simple terms.")

# print(response.content)


# import anthropic

# client = anthropic.Anthropic(
#     base_url="https://api.openmodel.ai",
#     api_key="om-WDypCJiTysfsM9ZvZeweNfB1Wob6d17YnZDy4BhTH",
# )

# message = client.messages.create(
#     model="deepseek-v4-flash",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Hello, who are you?"}
#     ],
# )
# print(message.content[0].text)

from openai import OpenAI

client = OpenAI(
    base_url="https://api.openmodel.ai/v1",
    api_key="om-WDypCJiTysfsM9ZvZeweNfB1Wob6d17YnZDy4BhTH",
)

response = client.responses.create(
    model="deepseek-chat",
    input="Hello, who are you?",
)
print(response.output_text)