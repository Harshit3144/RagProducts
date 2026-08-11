import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = TextLoader("document loaders/notes.txt")
docs=data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"),
    ("user","{data}")]
)

model = ChatMistralAI(
    model = "mistral-small-2603",
    api_key = os.getenv("MISTRAL_API_TOKEN")
)

prompt = template.format_messages(data=docs[0].page_content)
response = model.invoke(prompt)

print(response.content)