from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
  model='text-embedding-3-large',
  dimensions=32
)

documents = [
  "Dhaka is the capital of Bangladesh",
  "AI engineering is an emerging field",
  "AIs can not replace human engineers"
]

# it will generate an embedding of the given documents
result = embedding.embed_documents(documents)

print(str(result))