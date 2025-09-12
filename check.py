import os
import requests
from dotenv import load_dotenv
from crawling import get_content

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
print(GOOGLE_API_KEY)
print(GOOGLE_CX)

keyword = ["자동차 소음"]

all_results = []
filtered_results = []

google_url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
response = requests.get(google_url)
# dict_keys(['kind', 'title', 'htmlTitle', 'link', 'displayLink', 'snippet', 'htmlSnippet', 'formattedUrl', 'htmlFormattedUrl', 'pagemap'])

if response.status_code == 200:
    data = response.json()

for item in data['items']:
    title, content = get_content(item)
    print(item['link'])
    print(item['title'])
    print(content)

print(f"{len(filtered_results)} 개 결과 존재")