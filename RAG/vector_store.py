import json
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS

# print("PyTorch GPU 사용 가능:", torch.cuda.is_available())
# print("GPU 개수:", torch.cuda.device_count())
# print("현재 GPU 인덱스:", torch.cuda.current_device())
# print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))

device = "cuda" if torch.cuda.is_available() else "cpu"

file_path = './naver_blog_results_에어백 모듈 경고등 점등.json'
db_path = './faiss_db'
documents = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko", model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})
# embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", model_kwargs={'device':'cuda'}, encode_kwargs={'normalize_embeddings':True})

with open(file_path, 'r', encoding='utf-8') as f:
    blog = json.load(f)

for data in blog:

    splits = splitter.split_text(data['content']) # 리스트 안에 쪼개진 문자열들
    
    for split in splits:
        document = Document(
            page_content=split,
            metadata={
                "title": data['title'],
                "type":  data['type'],
                "url":  data['출처'],
                "car_type":  data['차종'],
                "engine_type": data['엔진'],
            }
        )
        documents.append(document)

vector_store = FAISS.from_documents(documents, embedding_model)
vector_store.save_local(db_path)