import nltk, re, pprint
from nltk import word_tokenize
from urllib import request

# change the path to where the nltk data is being stored
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
print(type(raw))
print(len(raw))
print(raw[:75])

# for language processing, we want to break into words and punctuation
# this is called tokenization
tokens = word_tokenize(raw)
print(type(tokens))
print(len(tokens))
print(tokens[:10])

# create NLTK text from this that we can do linguistic processing on
text = nltk.Text(tokens)
print(type(text))
print(text[1024:1062])
print(text.collocations())

# sometimes there is junk at the beginning and end of a text
# we will slice
raw = raw[raw.find("PART I"):raw.rfind("End of Project Gutenberg's Crime")]
print(raw.find("PART I"))

# dealing with html
url = 'http://news.bbc.co.uk/2/hi/health/2284783.stm'
html = request.urlopen(url).read().decode('utf8')
# use beautiful soup to get text out of html
from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
tokens = word_tokenize(raw)
tokens = tokens[110:390]
text = nltk.Text(tokens)
print(text.concordance('gene'))

# reading local files
"""
f = open('document.txt')
raw = f.read()
for line in f:
    print(line.strip())
"""
