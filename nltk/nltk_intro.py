import nltk

# change the path to where the nltk data is being stored
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

# import the book examples
from nltk.book import *

# in the select texts, find all instances of a certain word given context
print(text1.concordance('monstrous'))
print(text2.concordance('affection'))

# we can see what other words appear in similar contexts to a given word
print(text1.similar('monstrous'))
print(text2.similar('monstrous'))
# observe different results for different texts

# common contexts allow us to examine the contexts shared by two or more words
print(text2.common_contexts(['monstrous', 'very']))

# a dispersion plot is used to determine the location of a word in the text
# ie how many words from the beginning it appears
"""
print(text4.dispersion_plot(['citizens', 'democracy', 'freedom', 'duties', 'America']))
"""

# we can generate random text in certain styles
# this does not work atm
"""
print(text3.generate())
"""

# count the length of a text, punctuation included
print(len(text3))

# a little differently if we want the range of vocabulary or tokens used
"""
print(sorted(set(text3)))
"""
print(len(set(text3)))

# we can calculate the lexical richness of the text
# number of distinct words is % of total number of words
print(len(set(text3)) / len(text3))

# focusing on particular words, count how many times
# a word occurs in a text
print(text3.count('smote'))
print(100 * text4.count('a') / len(text4))

# we can wrap these in function form
def lexical_diversity(text):
    return len(set(text)) / len(text)
print(lexical_diversity(text3))

# we can think of a text like this
sent1 = ['Call', 'me', 'Bob', '.']
print(len(sent1))
print(lexical_diversity(sent1))

# check out the starting sentences of other texts
print(sent2)
print(sent3)

# finding a single word
print(text4[173])
"""
print(text4.index["awaken"])
# using index will find the first time the word appears
"""

# slicing
print(text5[16715:16735])

"""
saying = ['After', 'all', 'is', 'said', 'and', 'done', 'more', 'is', 'said', 'than', 'done']
tokens = set(saying)
tokens = sorted(tokens)
tokens[-2:]
"""

# frequency distributions
# find the top 50 most used words
fdist1 = FreqDist(text1)
print(fdist1)
print(fdist1.most_common(50))
print(fdist1['whale'])
"""
fdist1.plot(50, cumulative=True)
"""

# maybe we want words that are more than 15 characters long
V = set(text1)
long_words = [w for w in V if len(w) > 15]
print(sorted(long_words))

# we can introduce another parameter as well
# longer than 7 letters and appears more than 7 times
fdist5 = FreqDist(text5)
extra_parameter = [w for w in set(text5) if len(w) > 7 and fdist5[w] > 7]
print(sorted(extra_parameter))


# a collocation is a sequence of words that occur together unusually often
print(text4.collocations())
print(text8.collocations())

# look at the distribution of word lengths in a text
fdist = FreqDist(len(w) for w in text1)
print(fdist)
print(fdist.most_common())
print(fdist.max())
print(fdist[3])
print(fdist.freq(3))
"""
fdist.plot(50, cumulative=False)
"""

# a couple word comparison operators
print(sorted(w for w in set(text1) if w.endswith('ableness')))
print(sorted(term for term in set(text4) if 'gnt' in term))
print(sorted(item for item in set(text6) if item.istitle()))
print(sorted(item for item in set(sent7) if item.isdigit()))

# complex conditionals
print(sorted(w for w in set(text7) if '-' in w and 'index' in w))
print(sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10))
print(sorted(w for w in set(sent7) if not w.islower()))
print(sorted(t for t in set(text2) if 'cie' in t or 'cei' in t))

# wipe The and the difference
print(len(set(text1)))
print(len(set(word.lower() for word in text1)))
# filter out nonalphabetic items too
print(len(set(word.lower() for word in text1 if word.isalpha())))
