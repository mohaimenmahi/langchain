from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-opus-5')
result = model.invoke("Why so serious")

print(result.content)