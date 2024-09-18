from pymongo import MongoClient
import pandas as pd
import numpy as np
from numpy import array
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


client = MongoClient("localhost:27017")
db=client.hotel_reviews
result=db.Sentiment_Reviews.find({})
source=list(result)
df=pd.DataFrame(source)
df=df[["review","label"]]
df.head()


# Drop null values, ensure none left
df = df.dropna(how='any',axis=0)
df.isnull().values.any()


df=df.sample(100000,random_state=1)

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

X = []
sentences = list(df['review'])
for sen in sentences:
    X.append(preprocess_text(sen))

y = df['label']

y = np.array(list(map(lambda x: 1 if x=="POSITIVE" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ------------------------------------------------------
# Tokenizing
# ------------------------------------------------------

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from matplotlib import pyplot as plt


max_words=5000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = 100  

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1


# ------------------------------------------------------
# Convolutional NN (1m49s with 100k samples)
# ------------------------------------------------------

from keras.layers.convolutional import Conv1D
from keras.layers import GlobalAveragePooling1D

# create the model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=maxlen))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

# Model fitting
history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# --------------------------------------------------
# Standard model
# --------------------------------------------------

# create the model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=maxlen))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

# Fit the model
history=model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Final evaluation of the model
score = model.evaluate(X_test, y_test, verbose=1)

print(model.metrics_names)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()