import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_TOKEN")
print("Loaded key:", api_key)  # Debug check

# Use a proper chat model
model = ChatMistralAI(
    model="mistral-small-2402",   # ✅ change to a chat model
    api_key=api_key
)

parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template("Explain {topic} in short way")
detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detailed way")

topic = "troy war & odyssey"

# Parallel chain
chain = RunnableParallel({
    "short": short_prompt | model | parser,
    "detailed": detailed_prompt | model | parser
})

response = chain.invoke({"topic": topic})

print("Short:", response["short"])
print("Detailed:", response["detailed"])
