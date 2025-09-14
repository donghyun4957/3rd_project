import os
import sys
import urllib.request
from dotenv import load_dotenv
from crawling import get_kin_content
import json


# 키워드 json 파일로 가져오기 → 리스트로 뽑기
keyword_path = './common_keyword.txt'

with open(keyword_path, "r", encoding='utf-8') as f:
    common_keyword = json.load(f)

print(common_keyword)


# 블로그 링크만 따로 크롤링해서 가져오기
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


# 딕셔너리 생성
results = {}

# 위에서 가져온 키워드 리스트 for문 돌려서 크롤링(일단 100개)
for kw in common_keyword:
    contents = []
    seen_links = set()

    for start in range(1, 100, 10):
        encText = urllib.parse.quote(kw)
        url = f"https://openapi.naver.com/v1/search/kin.json?query={encText}&display=10&start={start}"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if(rescode==200):
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))


            for item in result["items"]:
                link = item.get('link')
                if not link or link in seen_links:
                    continue
                qna_title_text, qna_question_text, answer_texts = get_kin_content(link)
                if answer_texts != "내용 없음":
                        answer_dict = {}
                        for i, ans in enumerate(answer_texts, 1):
                            key = f"답변 {i}"
                            answer_dict[key] = ans
                        
                        content = {'질문': qna_question_text, '답변': answer_dict, '제목': qna_title_text, '링크': link}
                        
                        contents.append(content)
                        seen_links.add(link)

                # links = [item["link"] for item in result["items"]]
                results[kw] = contents

        else:
            print("Error Code:" + rescode)

    print(f"키워드 {kw}에는 총 {len(contents)} 개 결과 존재")


# json 파일로 저장하기
with open("naver_kin_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("JSON 파일로 저장 완료")