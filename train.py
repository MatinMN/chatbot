import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import random
import json
from tensorflow import keras

stemmer = LancasterStemmer()
nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])


words = [stemmer.stem(w.lower()) for w in words] #stem the words
words = sorted(list(set(words))) # remove duplicate words and sort them

labels = sorted(labels)


training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1 

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

model = keras.Sequential([
    keras.layers.Input(len(training[0])),
    keras.layers.Dense(8,activation="relu"),
    keras.layers.Dense(8,activation="relu"),
    keras.layers.Dense(len(output[0]),activation="softmax"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 

model.fit(training,output,epochs=200,batch_size=8,verbose=1)

model.save("model.h5")

results = model.evaluate(training,output)

print(results)