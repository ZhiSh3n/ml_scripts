import nltk
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

# stopwords are high frequency words that we want to
# filter out of a document before processing
# this is because they have little lexical content
# their appearance in a text does not distinguish it from
# other texts

from nltk.corpus import stopwords
print(stopwords.words('english'))

# define a function to compute what fraction of words
# in a text are not in the stopwords list
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)

# test this on reuters
print(content_fraction(nltk.corpus.reuters.words()))
