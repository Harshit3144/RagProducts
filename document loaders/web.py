from langchain_community.document_loaders import WebBaseLoader

url="https://en.wikipedia.org/wiki/History_of_India#Prehistoric_era_(before_c._3300_BCE)"

data = WebBaseLoader(url)

docs = data.load()

print(docs)