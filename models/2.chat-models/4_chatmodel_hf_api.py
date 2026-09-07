from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='HuggingFaceH4/zephyr-7b-beta',
  task='text-generation',
  provider='featherless-ai'
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Tell me something about backpropagation")

print(result.content)