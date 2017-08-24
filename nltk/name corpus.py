import nltk

# change the path to where the nltk data is being stored
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

names = nltk.corpus.names

# finding ambiguous names
print(names.fileids())
male_names = names.words('male.txt')
female_names = names.words('female.txt')
to_print = [w for w in male_names if w in female_names]
print(to_print)

# see that names ending in a are almost always female
cfd = nltk.ConditionalFreqDist(
    (fileid, name[-1]) # for the last letter
    for fileid in names.fileids()
    for name in names.words(fileid))
cfd.plot()
