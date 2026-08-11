import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParsers

search_tool = TavilySearchResults(max_results=5)

llm=ChatMistralAI(
    model="mistral-small-2402", 
    api_key=os.getenv("MISTRAL_API_TOKEN"))

prompt=ChatPromptTemplate.from_template(
    """
you are a helpful assistant. use the following above news and summarize the bulletin points{news}"""
) 

chain = prompt | llm | StrOutputParsers()

news_result = search_tool.run("latest news on ai")

result = chain.invoke({"news": news_result})

print(result.content)