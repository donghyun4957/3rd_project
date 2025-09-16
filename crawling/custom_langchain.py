import os
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def refine_text(model, content):

    instruction = """
    당신은 자동차 고장진단 전문가입니다. 
    아래 긴 원시 텍스트를 읽고, 각 정보들을 구조화하여 정리해 주세요. 

    출력 형식은 반드시 JSON이어야 하며, 절대 다른 문장으로 작성하지 마세요.

    원시 텍스트들은 카테고리별 증상, 원인 혹은 조치사항 등의 내용이 서로 붙어있습니다.
    각 증상별로 구분지어 구조화시키되, 글자를 요약, 변경 없이 그대로 나누세요
    하나의 증상에 다수의 내용들이 있으니 내용별로 리스트로 만들어주세요.
    주어진 텍스트가 증상, 원인, 조치사항 등의 카테고리 중 어디어 속할지 잘 판단하세요.
    그리고 원인과 조치사항들이 어떤 증상에 같이 포함되어야할지 잘 판단하세요.

   주의:
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