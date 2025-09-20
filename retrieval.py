import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


store = {}

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


def get_by_session_id(session_id):
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


def make_chain(model):

    instruction = """
    당신은 자동차 전문가 챗봇입니다. 사용자 질문과 제공된 문서 내용을 참고하여
    최적의 답변을 생성하세요.

    출력 지침:
    1. 답변은 마크다운 표나 불필요한 기호 없이 자연스러운 문단 형식으로 작성하세요.
    2. 제목이나 소제목은 간단히 정리하되, ###, --- 같은 구분선은 사용하지 마세요.
    3. 글머리표가 필요하면 • 나 - 같은 단순한 목록 기호만 사용하세요.
    4. 설명은 운전자나 일반 독자가 이해하기 쉽게 풀어서 작성하세요.
    6. 전체적으로 자연스럽고 정돈된 블로그 글이나 정비소 설명처럼 작성하세요. 

    단계별 사고(Chain of Thought):
    1. 질문을 이해하고, 제공된 문서 내용을 분석합니다.
    2. 분석한 문서 내용과 이해한 질문을 바탕으로 답변을 생성합니다.
    5. 최종 답변은 자연스럽고 읽기 쉽게 작성합니다.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction),
        MessagesPlaceholder(variable_name='history'),
        HumanMessagePromptTemplate.from_template(
            "사용자 질문: {query}\n\n"
            "참고 문서:\n{content}\n\n"
            "위 질문과 문서를 바탕으로 답변을 작성해주세요."
        )
    ])
    
    chain = prompt | model | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_by_session_id,
        input_messages_key='query',
        history_messages_key='history'
    )

    return chain_with_history

def make_filter(filter: dict):

    if any(list(filter.values())):
        main_filter = filter.copy()
    else:
        main_filter = None

    sub_filter = {k: {"$eq": "any"} for k in filter.keys()}

    return main_filter, sub_filter


if __name__ == "__main__":

    load_dotenv()
    db_path = './database/faiss_db'
    HF_TOKEN = os.getenv('HF_TOKEN')

    embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko")

    endpoint = HuggingFaceEndpoint(
        # repo_id='openai/gpt-oss-20b',
        repo_id='google/gemma-7b-it',
        task='text-generation',
        max_new_tokens=1024,
        huggingfacehub_api_token=HF_TOKEN,
    )

    model = ChatHuggingFace(llm=endpoint, verbose=True)
    
    # model_name = "yanolja/YanoljaNEXT-EEVE-Instruct-10.8B"

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    # llm = HuggingFacePipeline(pipeline=pipe)

    chain = make_chain(model)

    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

    query = "자동차가 브레이크가 밀리는데 원인이 뭐야?"
    filter = {"차종": None, "엔진": None}

    main_filter, sub_filter = make_filter(filter)

    if main_filter:
        main_docs = vector_store.similarity_search(query, k=3, filter=main_filter)
    else:
        main_docs = []
    print('main_docs: ', main_docs)

    sub_docs = vector_store.similarity_search(query, k=3, filter=sub_filter)
    
    print('sub_doc: ', sub_docs)

    # for res in sub_docs:
    #     print(f"* {res.page_content} [{res.metadata}]")

    context_text = '\n'.join([doc.page_content for doc in main_docs + sub_docs])

    response = chain.invoke({'query': query, 'content': context_text}, config={'configurable': {'session_id': 'user'}})
    
    print(response)