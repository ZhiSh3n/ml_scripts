import nltk
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

# project gutenberg contains many free electronic books
print(nltk.corpus.gutenberg.fileids())

# pick a text and see how long it is
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
print(len(emma))

# remember concordance from the intro?
emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
"""
print(emma.concordance('surprize'))
"""

# we could also just do
from nltk.corpus import gutenberg
print(gutenberg.fileids())
emmat = gutenberg.words('austen-emma.txt')

# quick intro analysis
# outputs average word length, average sentence length, average number of times vocab appears in the text
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid)) # counts spaces too, so av word length should be -1
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
    
# getting the longest sentence in a text
macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
longest_len = max(len(s) for s in macbeth_sentences)
print([s for s in macbeth_sentences if len(s) == longest_len])

# how about web chat?
from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')

# another web chat example
from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
print(chatroom[123])

# examine the brown corpus
from nltk.corpus import brown
print(brown.categories())
print(brown.words(categories='news'))
"""
print(brown.words(fileids=['cg22']))
print(brown.sents(categories=['news', 'editorial', 'reviews']))
"""

# produce counts for a particular genre
news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m], end = ' ')


# now obtain counts for each genre of interest 
cfd = nltk.ConditionalFreqDist((genre, word)
                               for genre in brown.categories()
                               for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)
# note that the most frequent modeal in romance is could
# the most frequent modal in news is will

# inaugural address corpus
from nltk.corpus import inaugural
print(inaugural.fileids())

# getting each year
print([fileid[:4] for fileid in inaugural.fileids()])

# look at how america and citizen are used over time
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
"""
cfd.plot()
"""

# examining length differences in translated languages
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut',
             'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)

# plot frequency distribution of the letters
raw_text = udhr.raw('English-Latin1')
nltk.FreqDist(raw_text).plot()

# the basic functions of nltk are raw, words, and sents

"""
# loading your own corpus
from nltk.corpus import PlaintextCorpurReader
corpus_root = '/data'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
print(wordlists.fileids()) # reads all the file names
print(wordlists.words('filename')) # prints words in file called filename
"""

# conditional frequency distributions
# counting words by genre
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genre_word = [(genre, word)
              for genre in ['news', 'romance']
              for word in brown.words(categories=genre)]
print(len(genre_word))
cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd)
print(cfd['news'])
print(cfd['romance'])
print(cfd['romance'].most_common(20))

# limited tabulation
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
             'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions=['English', 'German_Deutsch'],
             samples=range(10), cumulative=True)


