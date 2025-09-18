import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def make_chain(model):

    instruction = """

    """

    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction + "\n\n출력은 반드시 아래 형식 지침을 따르세요:\n{format_instructions}"),
        HumanMessagePromptTemplate.from_template("아래의 원문을 지시사항에 맞게 정리해주세요\n\n원문:\n{content}")
    ])
    
    chain = prompt | model | json_parser

    return chain, format_instructions

if __name__ == "__main__":

    load_dotenv()
    db_path = './chroma_db'
    HF_TOKEN = os.getenv('HF_TOKEN')

    embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    endpoint = HuggingFaceEndpoint(
        repo_id='openai/gpt-oss-20b',
        task='text-generation',
        max_new_tokens=1024,
        huggingfacehub_api_token=HF_TOKEN
    )

    model = ChatHuggingFace(llm=endpoint, verbose=True)
    chain = make_chain(model)

    vector_store = FAISS.load_local(db_path, embedding_model, collection_name="car_db", allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'kj':3})

    query = "왕비가 백설공주에게 먹인 것은 무엇인가요?"

    # chain of thought
    # 메타 데이터용 정보 추출
    # 메타 데이터 필터링
    retrievals = retriever.batch([query])
    context_text = '\n'.join([doc.page_content for doc in retrievals[0]])

    response = chain.invoke({'query': query, 'context': context_text})