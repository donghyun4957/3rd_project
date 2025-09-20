import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.model_laboratory import ModelLaboratory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

endpoint = HuggingFaceEndpoint(
    repo_id='MLP-KTLim/llama-3-Korean-Bllossom-8B',
    task='text-generation',
    max_new_tokens=1024,
    huggingfacehub_api_token=HF_TOKEN
)

hf_model = ChatHuggingFace(
    llm=endpoint,
    verbose=True
)

model = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0.1,
    max_tokens=2048
)

hf_model.invoke('아침으로 사과를 먹으면 좋을까?')

model_lab = ModelLaboratory.from_llms([model, hf_model])
model_lab.compare('아침에 사과를 먹는 것의 효과를 알려줘')