import re
import urllib.request
import requests
from pathlib import Path

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

url_stem = 'https://www.dictionary.com/browse/'
audio_regex = r'https://nonprod-audio.dictionary.com.+?\.mp3'
word = 'repine'

def get_pronunciation(word):
    word = word.lower()
    file_path = 'pronunciations/'+word+'.mp3'
    if Path(file_path).is_file():
        return
    opener = AppURLopener()
    try:
        response = opener.open(url_stem+word)
    except urllib.error.HTTPError:
        return 
    except UnicodeEncodeError:
        return
    try:
        html = response.read().decode('utf-8')
    except ValueError:
        return
    m = re.search(audio_regex, html)
    if m:
        print(m.group(0))
        r = requests.get(m.group(0))
        with open(file_path, 'wb') as f:
            f.write(r.content)

with open('cmudict-0.7b', encoding="ISO-8859-1") as f:
    count = 0
    skip = 104104
    for line in f:
        if not line.startswith(';;;'):
            print(count, line)
            if count >= skip:
                get_pronunciation(line.split('  ')[0])
            count += 1
