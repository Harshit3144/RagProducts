#dependencies installations.
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print
from langchain.agents import create_agent

#weather
@tool
def get_weather(city: str) -> str:
    """
    Returns the current weather for a given city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Unable to fetch weather data')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"The current temperature in {city} is {temp}°C with {desc}."

#news
@tool
def get_news(topic: str) -> str:
    """
    Returns the latest headlines for a given topic
    """
    api_key = os.getenv("NEWS_API_KEY")  # store your NewsAPI key in env variable
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}&language=en&pageSize=3"

    response = requests.get(url)
    data = response.json()

    print("DEBUG", data)

    if data.get("status") != "ok":
        return f"Error: {data.get('message', 'Unable to fetch news data')}"

    articles = data.get("articles", [])
    if not articles:
        return f"No news found for {topic}."

    headlines = []
    for article in articles:
        headlines.append(f"- {article['title']} ({article['source']['name']})")

    return f"Latest news on {topic}:\n" + "\n".join(headlines)






# Agent loop(impt)

llm = ChatMistralAI(
    model="mistral-small-2402",
    api_key=os.getenv("MISTRAL_API_TOKEN")
)

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful assistant that can provide weather updates and news headlines. Use the tools provided to fetch the required information."
)

print("city agent | type 'exit' to quit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print("bot:",response['messages'][-1].content)
