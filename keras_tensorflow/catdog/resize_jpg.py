import os

non_zero_cat_counter = 0
non_zero_dog_counter = 0

from PIL import Image

# lists to store the picture names
cat_list = []
dog_list = []

for file in os.listdir('data/dogs/'):
    if file.endswith('.jpg'):
        statinfo = os.stat('data/dogs/'+file)
        if statinfo.st_size > 0 :
            dog_list.append(file)
            im = Image.open('data/dogs/'+file)
            width, height = im.size
            non_zero_dog_counter = non_zero_dog_counter + 1
            if width != 200 or height != 200 :
                print('dogs' + file)
        else :
            os.remove('data/dogs/'+file)

for file in os.listdir('data/cats/'):
    if file.endswith(".jpg"):
        statinfo = os.stat('data/cats/'+file)
        if statinfo.st_size > 0 :
            cat_list.append(file)
            im = Image.open('data/cats/'+file)
            width, height = im.size
            non_zero_cat_counter = non_zero_cat_counter + 1
            if width != 200 or height != 200 :
                print('cats' +file)
        else :
            os.remove('data/cats/'+file)

print(non_zero_dog_counter)
print(non_zero_cat_counter)
print(len(cat_list))
print(len(dog_list))
