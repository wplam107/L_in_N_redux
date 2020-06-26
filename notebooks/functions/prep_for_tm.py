import pandas as pd
import pickle
import spacy

from nltk.corpus import stopwords

from clean_eng_funcs import load_articles, remove_unwanted, preprocess_for_bow

UNWANTED_HL = [
    'UPDATE',
    'US STOCKS',
    'PRESS']

UNWANTED_URL = [
    '/education/',
    '/politics/',
    '/diplomacy/',
    '/letters/',
    'health-',
    '/money/',
    '/transport/',
    'investing',
    '/society/']

stop_words = stopwords.words('english')
nlp = spacy.load('en', disable=['parser', 'ner'])

if __name__ == '__main__':
    # Load, clean, preprocess articles
    df = load_articles('articles.p')
    df = df.loc[(df['source'] != 'SCMP') & (df['date'] >= pd.Timestamp(2019, 3, 15))]
    df = remove_unwanted(df, 'headline', UNWANTED_HL)
    df = remove_unwanted(df, 'url', UNWANTED_URL)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)
    df = preprocess_for_bow(df, stop_words, nlp)

    # Create pickled DataFrame
    file = open('df_topic.p', 'wb')
    pickle.dump(df, file)
    file.close()

    