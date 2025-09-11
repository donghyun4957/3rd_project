import re
import requests
from bs4 import BeautifulSoup

def get_parser(url):
    response = requests.get(url)
    if response.status_code == 200:
        bs = BeautifulSoup(response.text, "html.parser")
    else: 
        print("HTTP 요청 실패;", response.status_code)
    return bs

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

def get_naver_content(url):
    
    base_url = "https://blog.naver.com"
    bs = get_parser(url)

    iframe = bs.find("iframe", {"id": "mainFrame"})
    if iframe:
        iframe_src = base_url + iframe["src"]
        bs2 = get_parser(iframe_src)
        content = bs2.find("div", class_="se-main-container")

    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"

def get_tistory_content(url):
    bs = get_parser(url)
    content = bs.find("div", class_="article") or bs.find("div", class_="tt_article_useless_p_margin")
    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음" 

def get_brunch_content(url):
    bs = get_parser(url)
    content = bs.find("div", class_="wrap_body")
    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"

def get_velog_content(url):
    bs = get_parser(url)
    content = bs.find("div", class_="sc-article-content") or bs.find("article")
    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"
    
def get_egloos_content(url):
    bs = get_parser(url)
    content = bs.find("div", id="entry_text")
    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"

def get_daum_blog_content(url):
    bs = get_parser(url)
    iframe = bs.find("iframe")
    if iframe:
        iframe_url = iframe["src"]
        bs2 = get_parser(iframe_url)
        content = bs2.find("div", class_="blogview_content")
    else:
        content = bs.find("div", class_="blogview_content")

    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"

def get_autospy_content(url):
    bs = get_parser(url)
    content = bs.find("div", class_="bbs-view-content") or bs.find("div", class_="board-content")
    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"

def get_encar_content(url):
    bs = get_parser(url)
    content = bs.find("div", class_="board__contents")
    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"

def get_bobaedream_content(url):
    bs = get_parser(url)
    content = bs.find("div", id="bbsContent")
    if content:
        result = content.get_text(" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', result)
        return clean_text
    else: 
        return "내용 없음"

def get_content(component):
    
    blog_url = component['link']
    blog_kind = component['displayLink'].lower()

    if "blog.naver.com" in blog_kind:
        return get_naver_content(blog_url)
    elif "tistory.com" in blog_kind:
        return get_tistory_content(blog_url)
    elif "brunch.co.kr" in blog_kind:
        return get_brunch_content(blog_url)
    elif "velog.io" in blog_kind:
        return get_velog_content(blog_url)
    elif "egloos.com" in blog_kind:
        return get_egloos_content(blog_url)
    elif "bobaedream.co.kr" in blog_kind:
        return get_bobaedream_content(blog_url)
    elif "encar.com" in blog_kind:
        return get_encar_content(blog_url)
    elif "autospy.net" in blog_kind:
        return get_autospy_content(blog_url)
    else:
        return "unknown"