from sklearn import tree
# tree allows us to categorize things in tree form

# these are the PROPERTIES of people
X = [[181,80,44], [177,70,43], [169,69,38], [154,54,37], [166,65,40],
     [190,90,47], [175,64,39], [177,70,40], [159,55,37], [171,75,42],
     [181,85,43]]

# these are the corresponding people's genders
Y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female', 'male']

classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(X,Y)

prediction = classifier.predict([190,70,43])

print(prediction)
