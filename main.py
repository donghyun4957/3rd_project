import os
import requests
from dotenv import load_dotenv
from crawling import get_content

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
print(GOOGLE_API_KEY)
print(GOOGLE_CX)

keyword = "자동차 소음"
all_results = []
filtered_results = []

for start in range(1, 100, 10):
    google_url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num=10&start={start}"
    response = requests.get(google_url)

    if response.status_code == 200:
        data = response.json()
        if "items" in data:
            all_results.extend(data["items"])
        else:
            break
print(f"총 {len(all_results)} 개 결과 수집 완료")

for item in all_results:
    content = get_content(item)
    if content != "내용 없음":
        filtered_results.append(item)
    else:
        print(item['link'])
        print(item['displayLink'])
        print(item['title'])

print(f"{len(filtered_results)} 개 결과 존재")