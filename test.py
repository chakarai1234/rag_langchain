import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name=model_name,
            **kwargs
        )


# 🔄 Replace with any OpenRouter model ID
model_id = "qwen/qwen2.5-vl-32b-instruct:free"

llm = ChatOpenRouter(model_name=model_id)

response = llm.stream("tell me 20 facts about Singapore")
for chunk in response:
    print(chunk.content, end="", flush=True)
