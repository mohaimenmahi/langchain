from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

# while running the code, first it will download the model 
# and it's dependencies locally, and will be loaded to the ram
llm = HuggingFacePipeline.from_model_id(
  model_id='HuggingFaceH4/zephyr-7b-beta',
  task='text-generation',
  pipeline_kwargs=dict(
    temperature=0.5,
    max_new_tokens=100
  )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Bangladesh")

print(result.content)