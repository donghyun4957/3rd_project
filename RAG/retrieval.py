import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
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
    HF_TOKEN = os.getenv('HF_TOKEN')

    endpoint = HuggingFaceEndpoint(
        repo_id='openai/gpt-oss-20b',
        task='text-generation',
        max_new_tokens=1024,
        huggingfacehub_api_token=HF_TOKEN
    )

    model = ChatHuggingFace(llm=endpoint, verbose=True)
    vector_store = FAISS.load_local('./db/faiss', text_embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    retrieval_qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type='stuff')

    retrieval_qa.invoke('인디언 조가 누구를 죽였나요?')