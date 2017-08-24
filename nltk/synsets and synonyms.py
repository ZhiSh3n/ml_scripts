import nltk

# change the path to where the nltk data is being stored
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

from nltk.corpus import wordnet as wn

# what is the syn set of motorcar?
print(wn.synsets('motorcar'))

# what are the lemmas of car then?
print(wn.synset('car.n.01').lemma_names())

# wn.synset('car.n.01').definition()
# wn.synset('car.n.01').examples()

# motor car is unambiguous, but car is
print(wn.synsets('car'))
for synset in wn.synsets('car'):
    print(synset.lemma_names())

# access all lemmas involving car
print(wn.lemmas('car'))
