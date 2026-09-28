from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chat_history = [
  SystemMessage(content='You are a helpful AI assistant')
]

print('Welcome to our dummy chatbot. Please enter \e to exit chat')
while True:
  user_input = input('You: ')
  if user_input == '\e':
    break
  chat_history.append(HumanMessage(content=user_input))
  result = model.invoke(chat_history)
  chat_history.append(AIMessage(content=result.content))
  print('AI: ', result.content)