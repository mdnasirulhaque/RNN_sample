
# coding: utf-8

# In[1]:


import nltk
import gensim
import numpy as np
import keras.backend as K
import logging

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# #Pretrained Model - Word Embedding - Word2Vec
# from nltk.data import find
# word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
# word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

# Using Goolgle Word2Vec model as the word embedding model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('W2V_Model/GoogleNews-vectors-negative300.bin/data', binary=True)


# In[3]:


# Importing Train and Test Data
import csv

emotions = ['anger', 
            # 'anticipation', 
            'disgust', 
            'fear', 
            'joy', 
            # 'love',
            'optimism', 
            'pessimism', 
            'sadness', 
            'surprise'] #, 
            # 'trust']

x_train_raw = []
y_train_raw = []
x_test_raw = []
y_test_raw = []


with open('Data/train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x_train_raw.append(row['Tweet'])
        y_train_raw_temp = []
        for emotion in emotions:
            y_train_raw_temp.append(row[emotion])
        y_train_raw.append(y_train_raw_temp)
        
with open('Data/test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x_test_raw.append(row['Tweet'])
        y_test_raw_temp = []
        for emotion in emotions:
            y_test_raw_temp.append(row[emotion])
        y_test_raw.append(y_test_raw_temp)
        
train_size = len(y_train_raw)
test_size = len(y_test_raw)

print("Train Size:", train_size, " samples")
print("Test Size:", test_size, " samples")


# In[4]:


SentiWords = {}
with open('Data/words.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        w = row['Words']
        SentiWords_temp = np.zeros(len(emotions), dtype=K.floatx())
        for i, emotion in enumerate(emotions):
            SentiWords_temp[i] = row[emotion]
        SentiWords[w] = SentiWords_temp
    SentiWords["QQ"] = np.zeros(len(emotions), dtype=K.floatx())


# In[5]:


# initialization Block
max_n_words = 40
ignore_words = ['?', '@', '-', '.', '_', '/', ' ', '.', '!',
               "you'll", 'itself', 'some', 'same', 'off', 'any', 'having',
                'and', 'theirs', 'your', 'should', 'after', 'out', 'in', 
                "you'd", 'd', 'its', 'had', 'myself','from', 'ourselves', 
                'here', 'an', 'all', 'yours', 'as', 'hers', 'they', 'll',
                "she's", 'through', 'you', 'then', 'once', 'my', 'am', 'who',
                'being', 'of', 'shan', 'that', 'so', 'with', 'yourselves',
                'both', 't', 'his', 'we', 'more', 'did', 'our', 'he', 'o', 
                'them', 'than', 'it', 'y', 'her', 'up', 'about', 'this', 
                'himself', 'just', 'if', 'own', 'has', 'how', 'because', 
                'him', 'doing', 'at', 'm', 'is', 'each', 's', 'too', 'those', 
                'such', 'have', 'above', "you've", 'most', 'on', 'under', 
                'by', 'few', 'where', 'when', 'were', "you're", "it's",
                'been', 'the', 'before', 'do', 'these', 'other', 'to', 'i',
                'can', 'themselves', 'what', 'are', 'while', 'which', 'me',
                'ma', "that'll", 've', 'for', 'why', 'a', 'during', 'yourself',
                'below', 'now', 'only', 'their', 'herself', 'will', 'does', 
                'she', 'be', 'there', "should've", 'was', 're', 'ours', 
                'whom', 'further']

# ignore_words = ['?', '@', '-', '.', '_', '/', ' ', '.', '!']

vector_size = len(word_embedding_model['I'])

print ("Vocabulary size: ",len(word_embedding_model.vocab), " words")

print ("Word-Vector size used: ", vector_size)


# In[6]:


# Cleaning and vectorizing the data

# Lemmatizing Functions
def get_wordnet_pos(tag):
    if (tag == ''):
        return ''
    elif tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return ''

def lemmatize_w(word, tag):
    wn_tag = get_wordnet_pos(tag)
    if (wn_tag == ''):
        return word
    else:
        return WordNetLemmatizer().lemmatize(word, wn_tag)

def lemmatize_s(sentence):
#     sent = []
#     word_tag = nltk.pos_tag(sentence)
    
#     for w, t in word_tag:
#         sent.append(lemmatize_w(w, t))
#     return sent
    return sentence


# Filtering the data
def filter_sent (sentence, embedding_model = word_embedding_model):
    
    # tokenize each word in the sentence
    s_words = word_tokenize(sentence.lower())
    
    filtered_sentence = []
    
    l = lemmatize_s(s_words)
    
    for w in l:
        # Remove words not in Vocab
        if w in embedding_model.vocab:
            filtered_sentence.append(w)
            
    return filtered_sentence

# Converting data to vectors
x_train = np.zeros((train_size, max_n_words, vector_size + len(emotions)), dtype=K.floatx())
y_train = np.zeros((train_size, len(emotions)), dtype=np.int32)
x_test = np.zeros((test_size, max_n_words, vector_size + len(emotions)), dtype=K.floatx())
y_test = np.zeros((test_size, len(emotions)), dtype=np.int32)

for i in range(train_size):
    x = filter_sent(x_train_raw[i])
    for index, word in enumerate(x):
        if word in SentiWords:
            vec = SentiWords[word]
        else:
            vec = SentiWords["QQ"]
        x_train[i, index, :] = np.concatenate((word_embedding_model[word], vec), axis=0)
        
for i, y in enumerate(y_train_raw):
    y_temp = np.zeros(len(emotions), dtype=np.int32)
    y_temp = np.array(y, dtype=np.int32)
    y_train[i, :] = y_temp
        
for i in range(test_size):
    x = filter_sent(x_test_raw[i])
    for index, word in enumerate(x):
        if word in SentiWords:
            vec = SentiWords[word]
        else:
            vec = SentiWords["QQ"]
        x_test[i, index, :] = np.concatenate((word_embedding_model[word], vec), axis=0)

        
for i, y in enumerate(y_test_raw):
    y_temp = np.zeros(len(emotions), dtype=np.int32)
    y_temp = np.array(y, dtype=np.int32)
    y_test[i, :] = y_temp
    
print ("Data Vectorized")

print ("Input shape: ", x_train.shape)
print ("Output shape: ", y_train.shape)


# In[7]:


from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing import sequence
from keras.layers import LSTM, Embedding

# Keras model
batch_size = 32  # 10
nb_epochs = 10   # 20

model = Sequential()

# RNN
model.add(LSTM(32, input_shape = (max_n_words, vector_size + len(emotions)), return_sequences = True, 
               dropout=0.1, recurrent_dropout=0.2))
model.add(LSTM(32, return_sequences=True)) 
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.2))

model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.1))

model.add(Dense(len(emotions), activation='sigmoid'))

# Compile the model
model.compile(loss = 'binary_crossentropy', 
              optimizer = RMSprop(lr=0.002, rho=0.9, epsilon=None, decay=1e-7),
              metrics = ['accuracy'])

print("MODEL:")
print(model.summary(), "\n")


# In[8]:


# Fit the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    shuffle=False, #False,
                    epochs=nb_epochs,
                    verbose=2,
                    validation_data=(x_test, y_test),
                    callbacks=[EarlyStopping(min_delta=1e-7, patience=3)])

# Fit the model (without early stop)
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     shuffle=False, #False
#                     epochs=nb_epochs,
#                     verbose=2,
#                     validation_data=(x_test, y_test))

print ("\n================================== Model Trained =================================\n")


# In[28]:


# Plotting

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
plt.show()


# In[10]:


# Classification
ERROR_THRESHOLD = 0.40

def classify(sentence, model, show_details=False):
    
    inp = np.zeros((train_size, max_n_words, vector_size + len(emotions)), dtype=K.floatx())
    
    filtered_sentence = filter_sent(sentence)
    x = filtered_sentence
        
    for index, word in enumerate(x):
        if word in SentiWords:
            vec = SentiWords[word]
        else:
            vec = SentiWords["QQ"]
        inp[0, index, :] = np.concatenate((word_embedding_model[word], vec), axis=0)
    
    results = model.predict(x = inp, batch_size=None)
    
    results2 = [[i,r] for i,r in enumerate(results[0]) if r > ERROR_THRESHOLD ] 
    results2.sort(key=lambda x: x[1], reverse=True)
    return_results = [[emotions[r[0]],r[1]] for r in results2]
    
    if show_details:
        print (emotions)
        print (results[0])
    
    if len(return_results) == 0:
    
    return return_results


# In[26]:


# Manual Testing
test_x = "I was fined for rash driving."
print (test_x, "\n")
print ("Prediction: ", classify(test_x, model), "\n")


# In[19]:


# Random manual Testing
n = np.random.randint(test_size - 1)
test_x = x_test_raw[n]

# n = np.random.randint(train_size - 1)
# test_x = x_train_raw[n]

print (test_x,"\n")

tags = []
for i, val in enumerate(y_test_raw[n]):
    if val == '1':
        tags.append(emotions[i])
print ("Actual Tag: ", tags, "\n")


print ("Prediction: ", classify(test_x, model), "\n")


# In[13]:


# Saving Model
# serialize model to JSON
model_json = model.to_json()
with open("Models/model_RNN_miltilabel.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("Models/model_RNN_miltilabel.h5")
print ("Saved model to disk")


# In[14]:


# Loading Model

from keras.models import model_from_json

# load json and create model
json_file = open('Models/model_RNN_miltilabel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("Models/model_RNN_miltilabel.h5")
print ("Loaded model from disk")


# In[15]:


# Evaluate loaded model on test data
loaded_model.compile(loss = 'binary_crossentropy', 
                     optimizer = RMSprop(lr = 0.002, rho = 0.9, epsilon = None, decay = 1e-7),
                     metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=1)
print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[17]:


# Manual Testing
test_x = "The truth is, life finds a way..."
print (test_x, "\n")
print ("Prediction: ", classify(test_x, loaded_model),"\n")

