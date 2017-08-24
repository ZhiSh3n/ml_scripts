import nltk

# change the path to where the nltk data is being stored
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

entries = nltk.corpus.cmudict.entries()

# find all words that ends with syllable sounding like nicks
syllable = ['N', 'IHO', 'K', 'S']
to_print = [word for word, pron in entries if pron[-4:] == syllable]
print(to_print)

# sounds like M but ends with n
to_print = [word for word, pron in entries if pron[-1] == 'M' and word[-1] == 'n']
print(to_print)

# stress patterns
def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]

to_print = [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
print(to_print)

to_print = [w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]
print(to_print)
