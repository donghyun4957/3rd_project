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

google_url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
response = requests.get(google_url)

if response.status_code == 200:
    data = response.json()

blog_url = data['items'][0]['link']
blog_kind = data['items'][0]['displayLink']

for component in data['items']:
    content = get_content(component)
    print('#################')
    print(component['displayLink'])
    print(content)