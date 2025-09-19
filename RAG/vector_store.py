import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS

# print("PyTorch GPU 사용 가능:", torch.cuda.is_available())
# print("GPU 개수:", torch.cuda.device_count())
# print("현재 GPU 인덱스:", torch.cuda.current_device())
# print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))

file_path = './test.json'
db_path = './faiss_db'
documents = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})

with open(file_path, 'r', encoding='utf-8') as f:
    blog = json.load(f)

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