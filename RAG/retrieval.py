import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory


class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        super().__init__()
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)

    def clear(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)


def make_chain(model):

    instruction = """
    당신은 자동차 전문가 챗봇입니다. 사용자 질문과 제공된 문서 내용을 참고하여
    최적의 답변을 생성하세요.

    단계별 사고(Chain of Thought):
    1. 질문을 이해하고, 제공된 문서 내용을 분석합니다.
    2. 문서에서 관련 정보를 찾아 요약하고 정리합니다.
    3. 차종/엔진 정보가 None이라도, 문서 내용을 기반으로 가능한 정확하게 답변합니다.
    4. 최종 답변은 자연스럽고 읽기 쉽게 작성합니다.
    """

    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction + "\n\n출력은 반드시 아래 형식 지침을 따르세요:\n{format_instructions}"),
        MessagesPlaceholder(variable_name='history'),
        HumanMessagePromptTemplate.from_template("아래의 원문과 지시사항을 참고하여 답변을 생성해주세요.\n\n원문:\n{content}")
    ])
    
    chain = prompt | model | json_parser

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_by_session_id,
        input_messages_key='query',
        history_messages_key='history'  # MessagesPlaceholder의 variable_name과 맞춰줘야 함
    )

    return chain_with_history


def get_by_session_id(session_id, store):
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


def extract_car_info(chain, query):
    """질문에서 차종/엔진 정보를 추출하는 간단한 CoT 체인"""
    extract_prompt = f"""
    질문: {query}

    1. 질문에서 차종과 엔진 정보를 확인하세요.
    2. 차종과 엔진 정보가 불분명하면, 다시 물어볼지 넘어갈지 판단하세요.
    2. 넘어가겠다고 판단하면 차종과 엔진 정보를 'Unknown'으로 표시하세요.

    출력:
    {{
        "차종": "...",
        "엔진": "..."
    }}
    """
    result = chain.invoke({'content': extract_prompt})
    car_type = result.get("차종", "Unknown")
    engine = result.get("엔진", "Unknown")
    return car_type, engine

if __name__ == "__main__":

    load_dotenv()
    db_path = './faiss_db'
    HF_TOKEN = os.getenv('HF_TOKEN')

    embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    endpoint = HuggingFaceEndpoint(
        repo_id='openai/gpt-oss-20b',
        task='text-generation',
        max_new_tokens=1024,
        huggingfacehub_api_token=HF_TOKEN
        model_kwargs={"device": "cuda"}
    )

    model = ChatHuggingFace(llm=endpoint, verbose=True)
    chain = make_chain(model)

    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})

    query = "자동차가 브레이크가 밀리는데 원인이 뭐야?"

    car_type, engine = extract_car_info(chain, query)

    retrievals = retriever.batch([query])
    context_text = '\n'.join([doc.page_content for doc in retrievals[0]])

    response = chain.invoke({'content': context_text})

    if car_type != "Unknown" and engine != "Unknown":
        k_docs_car = vector_store.similarity_search(
            query,
            k=3,
            filter={"차종": car_type, "엔진": engine}
        )
    else:
        k_docs_car = []

    k_docs_general = vector_store.similarity_search(
        query,
        k=3,
        filter={"차종": None, "엔진": None}
    )

    context_text = '\n'.join([doc.page_content for doc in k_docs_car + k_docs_general])

    response = chain.invoke({'content': context_text}, config={'configurable': {'session_id': 'user'}})

    print(response)