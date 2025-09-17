from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

text = " ".join(documents[4].page_content.split()[1:10])
vector = embedding_model.embed_query(text)

docs = [document.page_content for document in documents]
vects = embedding_model.embed_documents(docs)

text_embeddings = list(zip(docs, vects))
vector_store = FAISS.from_embeddings(text_embeddings, embedding_model)

vector_store.save_local('./db/faiss')
vector_db = FAISS.load_local(
    './db/faiss',
    text_embeddings,
    allow_dangerous_deserialization=True
)