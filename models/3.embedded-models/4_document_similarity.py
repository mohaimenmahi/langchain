from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

model = OpenAIEmbeddings(
  model='text-embedding-3-large',
  dimensions=300
)

documents = [
  "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
  "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
  "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
  "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
  "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about Virat Kohli"

doc_embeddings = model.embed_documents(documents)
query_embed = model.embed_query(query)

# print(cosine_similarity([query_embed], doc_embeddings))
# returns similarity score for each documents. We need to find the 
# highest similarity score.

scores = cosine_similarity([query_embed], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1] 
# getting the maximum score and it's index 

print(f"Query: {query}")
print(f"Answer: {documents[index]}")
print(f"Similarity score: {score}")