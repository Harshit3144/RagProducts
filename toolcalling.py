import os
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from rich import print 

#Creating a tool
@tool
def get_text_length(text:str)->str:
    """
    returns the number of characters in the text"""
    return len(text)

llm = ChatMistralAI(
    model="mistral-small-2402",
    api_key=os.getenv("MISTRAL_API_TOKEN")
)

#tools binding

llm_with_tool = llm.bind_tools([get_text_length])


result = llm_with_tool.invoke("return the length of the text 'hello how are you'")

print(result.content)
print(get_text_length.invoke({"text":"hello how are youz"}))