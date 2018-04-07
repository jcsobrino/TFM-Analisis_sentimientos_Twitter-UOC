import csv
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, strip_accents_unicode
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

stemmer = SnowballStemmer('spanish')
spanish_stopwords = stopwords.words('spanish')
analyzer = CountVectorizer().build_analyzer()

def read_corpus(filename):
    data = []
    label = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row[1])
            label.append(row[2])
    return data[:100], label[:100]

def preprocess_corpus(example, strip_accents=False, twitter_symbols=None):
    # convert to lowercase
    example = example.lower()
    # remove numbers, carriage returns and retweet old-style method
    example = re.sub(r'(\d+|\n|\brt\b)', '', example)
    # remove repeated characters
    example = re.sub(r'(.)\1+', r'\1', example) # r y l?

    if(strip_accents):
        # remove accents
        example = strip_accents_unicode(example)

    if twitter_symbols == 'remove':
        # remove mentions, hashtags and urls
        example = re.sub(r'(@|#|https?:)\S+', '', example)
    elif twitter_symbols == 'normalize':
        # normalize mentions, hashtags and urls
        example = re.sub(r'@\S+', 'user', example)
        example = re.sub(r'#\S+', 'hashtag', example)
        example = re.sub(r'https?:\S+', 'url', example)

    return example

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

data, label = read_corpus("datasets/global_dataset.csv")

scoring = {'accuracy': 'accuracy',
           'precision_macro': 'precision_macro',
           'recall_macro': 'recall_macro',
           'f1_macro': 'f1_macro',
           'precision_micro': 'precision_micro',
           'recall_micro': 'recall_micro',
           'f1_micro': 'f1_micro'}

pipeline = Pipeline([('vect', None),
                    ('clf', None)])
def p1(example):
    return preprocess_corpus(example, strip_accents=False, twitter_symbols='remove')

def p2(example):
    return preprocess_corpus(example, strip_accents=True, twitter_symbols='remove')

def p3(example):
    return preprocess_corpus(example, strip_accents=False, twitter_symbols='normalize')

def p4(example):
    return preprocess_corpus(example, strip_accents=True, twitter_symbols='normalize')

parameters = [
    {
        'vect': (TfidfVectorizer(),),
        'vect__preprocessor': (p1, p2, p3, p4),
        'vect__use_idf': (True, False),
        'vect__analyzer': ('word', stemmed_words),
        'vect__stop_words': (None, spanish_stopwords),
        'clf':(MultinomialNB(), KNeighborsClassifier(), DecisionTreeClassifier(), SVC())
    },
    {
        'vect': (CountVectorizer(),),
        'vect__preprocessor': (p1, p2, p3, p4),
        'vect__binary': (True, False),
        'vect__analyzer': ('word', stemmed_words),
        'vect__stop_words': (None, spanish_stopwords),
        'clf':(MultinomialNB(), KNeighborsClassifier(), DecisionTreeClassifier(), SVC())
    }
]
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=skf, verbose=2, scoring=scoring, refit='f1_micro')
    grid_search.fit(data, label)
    print("best_params:", grid_search.best_params_)
    print("best_score:", grid_search.best_score_)
    print(pd.DataFrame(grid_search.cv_results_).to_string())

