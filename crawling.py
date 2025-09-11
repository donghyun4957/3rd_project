import requests
from bs4 import BeautifulSoup

def get_parser(url):
    response = requests.get(url)
    if response.status_code == 200:
        bs = BeautifulSoup(response.text, "html.parser")
    else: 
        print("HTTP 요청 실패;", response.status_code)
    return bs

def get_blog_content(url):
    
    base_url = "https://blog.naver.com"
    bs = get_parser(url)

    iframe = bs.find("iframe", {"id": "mainFrame"})
    if iframe:
        iframe_src = base_url + iframe["src"]
        bs2 = get_parser(iframe_src)

        content = bs2.find("div", class_="se-main-container")

    result = content.get_text("\n", strip=True) if content else "내용 없음"
    
    return result

def get_kin_content(url):
    # soup = BeautifulSoup(response.content, 'html.parser')
    bs = get_parser(url)

    qna_title = bs.find("div", class_="endTitleSection")
    qna_title_text = qna_title.get_text(strip=True) if qna_title else "제목 없음"

    qna_question = bs.find("div", class_="questionDetail")
    qna_question_text = qna_question.get_text(" ", strip=True) if qna_question else "질문 없음"

    qna_answers = bs.find_all("div", class_="se-main-container")
    if qna_answers:
        answer_texts = [ans.get_text(" ", strip=True) for ans in qna_answers]
    else:
        answer_texts = ["답변 없음"]

    return qna_title_text, qna_question_text, answer_texts

if __name__ == "__main__":
    blog_url = "https://blog.naver.com/foxload51/223895820464"
    kin_url = "https://kin.naver.com/qna/detail.naver?d1id=8&dirId=8110301&docId=487019164&enc=utf8&kinsrch_src=pc_tab_kin&qb=7J6Q64+Z7LCoIOydtOyDge2VnCDshozrpqw%3D0"

    content = get_blog_content(blog_url)
    qna_title_text, qna_question_text, answer_texts = get_kin_content(kin_url)

    print("본문:", content)

    print("지식인 제목:", qna_title_text)
    print("지식인 질문:", qna_question_text)
    print("지식인 답변들:")
    for i, ans in enumerate(answer_texts, 1):
        print(f"{i}. {ans}")