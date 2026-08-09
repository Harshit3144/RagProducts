import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI


load_dotenv()

model = ChatMistralAI(model = "mistral-small-2603",
api_key = os.getenv("MISTRAL_API_TOKEN")
)

response = model.invoke("hii")

print(response)
