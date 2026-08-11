import os
from dotenv import load_dotenv


from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()
model = ChatMistralAI(
    model = "mistral-moderation-2603",
    api_key = os.getenv("MISTRAL_API_TOKEN")
)
parser = StrOutputParser()
#two differnt prompt
short_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in short way"
)
detailed_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in detailed way"
)

topic = "troy war & odessay"

chain = RunnableParallel({
    "short": RunnableLambda(lambda x:x['short'])|short_prompt | model | parser,
    "detailed": RunnableLambda(lambda x:x['detailed'])|detailed_prompt | model | parser
})

response = chain.invoke({
    "short": {"topic": "troy war & odessay"},
    "detailed": {"topic": "troy war & odessay"}
})

print(response["short"])
print(response["detailed"])
