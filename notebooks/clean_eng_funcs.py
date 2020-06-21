import pandas as pd
import pickle
import re
import gensim

from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
from gensim import corpora, models

def load_articles(filepath):
    """
    Load pickle file and output DataFrame
    """
    f = open(filepath, 'rb')
    df = pickle.load(f)
    f.close()
    return df

def remove_unwanted(df, column, phrases):
    """
    Remove articles with certain phrases in columns

    Parameters
    ----------
    df : DataFrame
    column : str
        DataFrame column name
    phrases : list
        list of phrases (str) to remove
    
    Returns
    -------
    df : DataFrame
    """
    for phrase in phrases:
        df = df.loc[df[column].map(lambda x: re.search(r'{}'.format(phrase), x)).isna()]
    return df

def _replace_words(text):
    text = re.sub(r'U\.S\.', 'US', text)
    text = re.sub(r'U\.S\.A\.', 'US', text)
    text = re.sub(r'US', 'USA', text)
    text = re.sub(r'U\.K\.', 'UK', text)
    text = re.sub(r'Mr\.', 'MR', text)
    text = re.sub(r'Mrs\.', 'MRS', text)
    text = re.sub(r'Ms\.', 'MS', text)
    text = re.sub(r'\.\.\.', '', text)
    text = re.sub(r'U.S-China', 'US-China', text)
    text = text.replace('Co.', 'Co')
    text = text.replace('\xa0', '')
    text = text.replace('."', '".')
    text = text.replace('immediatelywith', 'immediately with')
    text = text.replace('theOfficeof', 'the Office of')
    text = text.replace('theCommissionerof', 'the Commissioner of')
    return text

def _remove_repeats(sentences):
    original = []
    for sentence in sentences:
        if sentence not in original:
            original.append(sentence)
    return original

def _preprocess_sent(texts, stop_words):
    texts_out = []
    for text in texts:
        simple_text = gensim.utils.simple_preprocess(text)
        no_stop = [ word for word in simple_text if word not in stop_words ]
        texts_out.append(no_stop)
    return texts_out

def _preprocess_body(text, stop_words):
    simple_text = gensim.utils.simple_preprocess(text)
    text_out = [ word for word in simple_text if word not in stop_words ]
    return text_out

def _bigram_model(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def _make_bigrams_sent(texts, bigram_mod):
    return [ bigram_mod[doc] for doc in texts ]

def _make_bigrams(text, bigram_mod):
    return bigram_mod[text]

def _lemmatization(text, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(' '.join(text))
    return [ token.lemma_ for token in doc if token.pos_ in allowed_postags ]

def _lemmatize_sent(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(' '.join(sent))
        texts_out.append(' '.join([ token.lemma_ for token in doc if token.pos_ in allowed_postags ]))
    return texts_out

def preprocess_for_bow(df, stop_words, nlp):
    """
    Standardize certain phrases in text bodies, convert bodies to lists of sentences,
    create word tokens in sentences, runs bigrams, lemmatize word tokens

    Parameters
    ----------
    df : DataFrame
        DataFrame contains a body of text

    Returns
    -------
    df : DataFrame
        DataFrame returned has additional sentences, sentence_tokens, word_tokens columns
    """
    # Create new tokenized features
    df['body'] = df['body'].map(_replace_words)
    df['headline'] = df['headline'].map(_replace_words)
    df['sentences'] = df['body'].map(sent_tokenize)
    df['sentences'] = df['sentences'].map(_remove_repeats)
    df['sentence_tokens'] = df['sentences'].map(lambda x: _preprocess_sent(x, stop_words))
    df['word_tokens'] = df['sentence_tokens'].map(lambda x: [ item for l in x for item in l ])
    df['headline_tokens'] = df['headline'].map(lambda x: _preprocess_body(x, stop_words))
    
    # Create and apply bigram model
    data_words = df['body'].map(lambda x: _preprocess_body(x, stop_words))
    bigram_mod = _bigram_model(data_words)
    df['word_tokens'] = df['word_tokens'].map(lambda x: _make_bigrams(x, bigram_mod))
    df['sentence_tokens'] = df['sentence_tokens'].map(lambda x: _make_bigrams_sent(x, bigram_mod))
    df['headline_tokens'] = df['headline_tokens'].map(lambda x: _make_bigrams(x, bigram_mod))

    # Lemmatize tokens
    df['word_tokens'] = df['word_tokens'].map(lambda x: _lemmatization(x, nlp))
    df['sentence_tokens'] = df['sentence_tokens'].map(lambda x: _lemmatize_sent(x, nlp))
    df['headline_tokens'] = df['headline_tokens'].map(lambda x: _lemmatization(x, nlp))

    return df

def preprocess_for_we(df, stop_words, nlp):
    """
    Standardize certain phrases in text bodies, convert bodies to lists of sentences,
    create word tokens in sentences

    Parameters
    ----------
    df : DataFrame
        DataFrame contains a body of text

    Returns
    -------
    df : DataFrame
        DataFrame returned has additional sentences, sentence tokens, word tokens columns
    """
    # Create new tokenized features
    df['body'] = df['body'].map(_replace_words)
    df['headline'] = df['headline'].map(_replace_words)
    df['sentences'] = df['body'].map(sent_tokenize)
    df['sentences'] = df['sentences'].map(_remove_repeats)
    df['sentence_tokens'] = df['sentences'].map(lambda x: _preprocess_sent(x, stop_words))
    df['word_tokens'] = df['sentence_tokens'].map(lambda x: [ item for l in x for item in l ])
    df['headline_tokens'] = df['headline'].map(lambda x: _preprocess_body(x, stop_words))

    return df