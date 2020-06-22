import numpy as np
import pandas as pd
import pickle
import gensim
import nltk
import xgboost
from nltk.stem import WordNetLemmatizer 

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
xgb_model = pickle.load(open('models/xgb_model.p', 'rb'))
mallet = pickle.load(open('models/mallet.p', 'rb'))
id2word = pickle.load(open('models/id2word.p', 'rb'))
glove = pickle.load(open('models/glove.p', 'rb'))
lemmatizer = WordNetLemmatizer()

def _preprocess_body(text, stop_words=stop_words):
    simple_text = gensim.utils.simple_preprocess(text)
    text_out = [ word for word in simple_text if word not in stop_words ]
    return text_out

def _lemmatization(text, lemmatizer=lemmatizer):
    return [ lemmatizer.lemmatize(token) for token in text ]

def _preprocess(text, stop_words=stop_words):
    tokens = _preprocess_body(text)
    lemmed_tokens = _lemmatization(tokens)
    return tokens, lemmed_tokens

def _topics_in_doc(word_tokens, id2word=id2word, model=mallet):
    vec = id2word.doc2bow(word_tokens)
    return model[vec]

def _w2v(words, dictionary):
    vec = np.mean([ dictionary[word] for word in words if word in dictionary ], axis=0)
    return vec

def process_text(text):
    tokens, lemmed_tokens = _preprocess(text)
    topics = _topics_in_doc(lemmed_tokens) # In format ['protest', 'political', 'economic']
    word_vec = _w2v(tokens, glove)
    return topics, word_vec

def predict_source(word_vec, model=xgb_model):
    vec_df = pd.DataFrame(word_vec).T
    probs = model.predict_proba(vec_df) # In format ['ABC', 'CCTV', 'CNN', 'Reuters']
    return probs
    

