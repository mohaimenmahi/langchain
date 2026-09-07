from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
  model='text-embedding-3-large',
  dimensions=32
)

# it will generate an embedding of the given query
result = embedding.embed_query("Dhaka is the capital of Bangaldesh")

print(str(result))