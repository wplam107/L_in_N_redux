import numpy as numpy
import pandas as pd
import pickle
import gensim
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
nlp = spacy.load('en', disable=['parser', 'ner'])
xgb_model = pickle.load(open('models/xgb_model.p', 'rb'))
mallet = pickle.load(open('models/mallet.p', 'rb'))
id2word = pickle.load(open('models/id2word.p', 'rb'))
glove = pickle.load(open('models/glove.p', 'rb'))

def _preprocess_body(text, stop_words):
    simple_text = gensim.utils.simple_preprocess(text)
    text_out = [ word for word in simple_text if word not in stop_words ]
    return text_out

def _lemmatization(text, nlp=nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(' '.join(text))
    return [ token.lemma_ for token in doc if token.pos_ in allowed_postags ]

def _preprocess(text, stop_words=stop_words, nlp=nlp):
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
    topics = _topics_in_doc(lemmed_tokens)
    word_vec = _w2v(tokens, glove)
    return topics, word_vec

def predict_source(word_vec, model):
    vec_df = pd.DataFrame(word_vec).T
    probs = model.predict_proba(vec_df) # In format ['ABC', 'CCTV', 'CNN', 'Reuters']
    return probs
    

