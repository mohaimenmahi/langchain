from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.load import loads

load_dotenv()

model = ChatOpenAI()

st.header('Research Tool')

paper_input = st.selectbox(
  "Select Research Paper Name", [
    "Attention Is All You Need", 
    "BERT: Pre-training of Deep Bidirectional Transformers", 
    "GPT-3: Language Models are Few-Shot Learners", 
    "Diffusion Models Beat GANs on Image Synthesis"
  ]
)

style_input = st.selectbox(
  "Select Explanation Style", [
    "Beginner-Friendly", 
    "Technical", 
    "Code-Oriented", 
    "Mathematical"
  ] 
) 

length_input = st.selectbox(
  "Select Explanation Length", [
    "Short (1-2 paragraphs)", 
    "Medium (3-5 paragraphs)", 
    "Long (detailed explanation)"
  ] 
)

with open('template.json', 'r') as f:
  template = loads(f.read())

if st.button('Summarize'):
  chain = template | model # leveraging chain from langchain: template | model -> from template it gets the inputs and passes it to the model
  result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
  })
  st.write(result.content)