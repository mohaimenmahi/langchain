from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
  model_name='sentence-transformers/all-MiniLM-L6-v2',
)

text = "Dhaka is the capital of Bangladesh"

documents = [
  "Dhaka is the capital of Bangladesh",
  "AI engineering is an emerging field",
  "AIs can not replace human engineers"
]

vector = embedding.embed_query(text)
doc_vectors = embedding.embed_documents(documents)

print(str(vector))
print(str(doc_vectors))