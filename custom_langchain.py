import os
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def refine_text(model, content):

    instruction = """
    당신은 자동차 고장진단 전문가입니다. 
    아래 긴 텍스트를 읽고, 각 카테고리별로 관련 정보를 구조화하여 정리해 주세요. 
    각 카테고리별로 증상과 원인을 포함합니다:

    출력 형식은 반드시 JSON이어야 하며, 절대 다른 문장으로 작성하지 마세요.
    JSON 예시:
    {{
    "카테고리1": {{
        "증상1": ["원인1", "원인2"],
        "증상2": ["원인1"]
    }},
    "카테고리2": {{
        "증상1": ["원인1"]
    }}
    }}

    카테고리가 없으면 증상과 원인에 대한 딕셔너리로만 작성해도 괜찮습니다.
    하나의 증상에 다수의 원인이 있을 수 있으니 원인들은 리스트로 만들어주세요.

    주의:
    - 누락되는 내용 없이 가능한 모든 정보를 포함
    - 불필요한 문장은 제거
    - 항목별로 리스트로 나열
    """

    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction + "\n\n출력은 반드시 아래 형식 지침을 따르세요:\n{format_instructions}"),
        HumanMessagePromptTemplate.from_template("아래의 원문을 지시사항에 맞게 정리해주세요\n\n원문:\n{content}")
    ])
    
    chain = prompt | model | json_parser
    
    response = chain.invoke({
        "content": content,
        "format_instructions": format_instructions
    })

    return response