from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2603",
    api_key=os.getenv("MISTRAL_API_TOKEN")
)

system_msg = SystemMessage(
    content="Answer politely and ONLY using information from the provided document. If the answer is not in the document, say you cannot answer."
)

url = "https://en.wikipedia.org/wiki/History_of_India"
data = WebBaseLoader(url)
docs = data.load()

document_msg = HumanMessage(content=docs[0].page_content)

print("___Press 0 to exit___")
while True:
    user_input = input("You:")
    if user_input == "0":
        break
    response = model.invoke([system_msg, document_msg, HumanMessage(content=user_input)])
    print("\nBot:", response.content, "\n")
