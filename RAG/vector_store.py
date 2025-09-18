import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.vectorstores import FAISS


file_path = './naver_blog_results_tmp.json'
db_path = './chroma_db'
documents = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

with open(file_path, 'r', encoding='utf-8') as f:
    blog = json.load(f)
cnt = 0
for keys, vals in blog.items():
    for val in vals:
        splits = splitter.split_text(val['content']) # 리스트 안에 쪼개진 문자열들
        
        for split in splits:
            document = Document(
                page_content=split,
                metadata={
                    "title": val['title'],
                    "type":  val['type'],
                    "url":  val['출처'],
                    "car_type":  val['차종'],
                    "engine_type": val['엔진'],
                }
            )
            documents.append(document)

vector_store = FAISS.from_documents(documents, embedding_model)
vector_store.save_local(db_path)