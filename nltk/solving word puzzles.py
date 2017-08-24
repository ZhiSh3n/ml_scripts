import nltk

# change the path to where the nltk data is being stored
nltk.data.path.append('/Users/zhi/Documents/Programming/PROJECTS_Python/data')

puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
to_print = [w for w in wordlist if len(w) >= 6
            and obligatory in w
            and nltk.FreqDist(w) <= puzzle_letters]
# last conditional is to check that the frequency of each letter
# in the candidate word is less than or equal to the frequency
# of the corresponding letter in the puzzle
print(to_print)
