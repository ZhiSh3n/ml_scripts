import nltk
from nltk import word_tokenize
from urllib import request

# change the path to where the nltk data is being stored
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

# html
html = urlopen(url).read() # download web page
raw = nltk.clean_html(html) # strip remaining html
raw = raw[750:23506] # trim to desired content

# ascii
tokens = nltk.wordpunct_tokenize(raw) # tokenize the text
tokens = tokens[20:1834] # select tokens of interest
text = nltk.Text(tokens) # create nltk text

# vocab
words = [w.lower() for w in text] # normalize the words
vocab = sorted(set(words)) # build the vocabulary
