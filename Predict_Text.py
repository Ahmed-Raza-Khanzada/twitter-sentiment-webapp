import re
import string

# import tensorflow

import nltk

# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download("wordnet")
# from wordcloud import WordCloud,STOPWORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
stemmer= PorterStemmer()
lemmatizer=WordNetLemmatizer()

# import pickle
# from tensorflow.keras.models import load_model

# filepath="sentiment_model.h5"
# model1 = load_model(filepath)

# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer1= pickle.load(handle)
    
    
try:
    with open("train/stop_words.txt",'r') as stopword1:
        stopwords1 = stopword1.read()
        stopwords1 = stopwords1[1:-2].split(", ")
        stopwords = [word[1:-1] for word in stopwords1]
except:
    with open("stop_words.txt",'r') as stopword1:
        stopwords1 = stopword1.read()
        stopwords1 = stopwords1[1:-2].split(", ")
        stopwords = [word[1:-1] for word in stopwords1]

def lem_stem_stop(comms1):
    comments3 = []
    not_word = ["not","havent","hasnt","isnt","doesnt","dont","shouldnt","cant","wasnt","like"]
    for com in comms1:
        temp = []
        com=word_tokenize(com)
        for word in com:
          if word not in not_word: 
              s_word =stemmer.stem(word)
              l_word = lemmatizer.lemmatize(s_word)
              if l_word not in stopwords:
                  temp.append(l_word)
              else:
                  pass
          else:
              temp.append(word)
        comments3.append(" ".join(temp))
    return comments3    


def predict_text(text,model,tokenizer ,filepath="sentiment_model.h5"):
    l = []
    for d in text:
      document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
      document_test = re.sub(r'@\w+', '', document_test)
      document_test = document_test.lower()
      document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
      document_test = re.sub(r'[0-9]', '', document_test)
      document_test = re.sub(r'\s{2,}', ' ', document_test)
      l.append(document_test)
    text = lem_stem_stop(l)
    
    X = tokenizer.texts_to_sequences(text)

    X = pad_sequences(X,maxlen=35)
    # model = load_model(filepath)
    # print("Input ->", X)
    y = model.predict(X)
    labels = ['Positive' if x>=0.50 else 'Negative' for x in y ]
    confidence = [x if x>=0.50 else (1-x) for x in y ]
    return labels,confidence


    
# print(predict_text("Good", model1, tokenizer1))