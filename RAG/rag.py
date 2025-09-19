import os
import json
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

store = {}
metadata_store = {}

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

def get_session_metadata(session_id):
    if session_id not in metadata_store:
        metadata_store[session_id] = {}
    return metadata_store[session_id]

uncertainty_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "너는 자동차 전문가이며, 사용자 질문에 답하기 전에 필요한 정보(차종, 엔진타입 등)가 부족한지 판단한다."
    ),
    MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.from_template(
        """
        사용자 질문: {user_question}
        필요한 메타데이터 키: {metadata_keys}

        부족한 정보가 있으면 되물어라.
        JSON으로 출력:
        {{
        "requires_info": true/false,
        "missing_info": [list of metadata keys],
        "clarification_question": "string"
        }}
        """
    )
])

# 사용자 답변에서 key-value 추출
parse_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "사용자의 응답에서 부족한 메타데이터 키별 값을 추출해라."
    ),
    HumanMessagePromptTemplate.from_template(
        """
        필요한 키: {missing_keys}
        사용자 응답: {user_answer}

        JSON으로 출력:
        {{
            "parsed": {{
                "키1": "값1",
                "키2": "값2",
                ...
            }}
        }}
        """
    )
])

answer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "너는 자동차 전문가이며, 항상 대화 히스토리를 참고해 문맥을 이어간다."
    ),
    MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.from_template(
        """
        사용자 질문: {user_question}
        수집된 메타데이터: {metadata}
        검색된 문서: {docs}

        문서를 기반으로 자연스러운 답변을 제공하라.
        """
    )
])


load_dotenv()
db_path = './faiss_db'
HF_TOKEN = os.getenv('HF_TOKEN')

embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

endpoint = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    max_new_tokens=512,
    huggingfacehub_api_token=HF_TOKEN,
)

model = ChatHuggingFace(llm=endpoint, verbose=True)

uncertainty_chain = uncertainty_prompt | model
parse_metadata_chain = parse_prompt | model
answer_chain = answer_prompt | model

uncertainty_with_history = RunnableWithMessageHistory(
    uncertainty_chain,
    get_session_history=get_by_session_id,
    input_messages_key="user_question",
    history_messages_key="history",
)

parse_metadata_with_history = RunnableWithMessageHistory(
    parse_metadata_chain,
    get_session_history=get_by_session_id,
    input_messages_key="user_answer",
    history_messages_key="history",
)

answer_with_history = RunnableWithMessageHistory(
    answer_chain,
    get_session_history=get_by_session_id,
    input_messages_key="user_question",
    history_messages_key="history",
)


def clarification_loop(user_question, metadata_keys, session_id, max_attempts=2):
    collected_metadata = get_session_metadata(session_id)
    config = {"configurable": {"session_id": session_id}}

    for _ in range(max_attempts):
        needed_keys = [k for k in metadata_keys if k not in collected_metadata]
        if not needed_keys:
            break

        # 부족한 key 판단
        result = uncertainty_with_history.invoke(
            {"user_question": user_question, "metadata_keys": needed_keys},
            config=config
        )
        parsed = json.loads(result.content)

        if not parsed["requires_info"]:
            break

        user_answer = input(f"{parsed['clarification_question']}: ")

        # 사용자 응답에서 key-value 추출
        parse_result = parse_metadata_with_history.invoke(
            {"missing_keys": needed_keys, "user_answer": user_answer},
            config=config
        )
        parsed_dict = json.loads(parse_result.content).get("parsed", {})

        # 수집
        for key, value in parsed_dict.items():
            collected_metadata[key] = value

    # 남은 key는 None 처리
    for k in metadata_keys:
        if k not in collected_metadata:
            collected_metadata[k] = None

    return collected_metadata


def retrieve_documents(collected_metadata, vectorstore, query, k=5):
    filters = {k: v for k, v in collected_metadata.items() if v is not None}
    
    docs = vectorstore.similarity_search(query, k=k, filter=filters)
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
    context_text = '\n'.join([doc.page_content for doc in k_docs_general])

    return context_text


def rag_with_memory(user_question, metadata_keys, vectorstore, session_id="default"):
    collected_metadata = clarification_loop(user_question, metadata_keys, session_id)
    docs = retrieve_documents(collected_metadata, vectorstore, user_question)

    config = {"configurable": {"session_id": session_id}}
    answer = answer_with_history.invoke(
        {
            "user_question": user_question,
            "metadata": collected_metadata,
            "docs": docs
        },
        config=config
    )
    return answer.content

vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
metadata_keys = ["차종", "엔진타입"]

user_question = "자동차에서 이게 안되네, 왜그래?"
final_answer = rag_with_memory(user_question, metadata_keys, vectorstore, session_id="user")
print("최종 답변:", final_answer)
