import csv

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Preprocessor import Preprocessor

spanish_stopwords = stopwords.words('spanish')

def read_corpus(filename):
    data = []
    label = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row[1])
            label.append(row[2])
    return data[:1000], label[:1000]

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

parameters = [
    {
        'vect': (TfidfVectorizer(),),
        'vect__preprocessor': (Preprocessor(strip_accents=False, twitter_symbols='remove', stemming=False).preprocess,
                               Preprocessor(strip_accents=False, twitter_symbols='remove', stemming=True).preprocess,
                               Preprocessor(strip_accents=False, twitter_symbols='normalize', stemming=False).preprocess,
                               Preprocessor(strip_accents=False, twitter_symbols='normalize', stemming=True).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='remove', stemming=False).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='remove', stemming=True).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='normalize', stemming=False).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='normalize', stemming=True).preprocess),
        'vect__ngram_range': ((1,1), (1,2), (1,3)),
        'vect__use_idf': (True, False),
        'vect__stop_words': (None, spanish_stopwords),
        'clf':(MultinomialNB(), KNeighborsClassifier(), DecisionTreeClassifier(), SVC())
    },
    {
        'vect': (CountVectorizer(),),
        'vect__preprocessor': (Preprocessor(strip_accents=False, twitter_symbols='remove', stemming=False).preprocess,
                               Preprocessor(strip_accents=False, twitter_symbols='remove', stemming=True).preprocess,
                               Preprocessor(strip_accents=False, twitter_symbols='normalize', stemming=False).preprocess,
                               Preprocessor(strip_accents=False, twitter_symbols='normalize', stemming=True).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='remove', stemming=False).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='remove', stemming=True).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='normalize', stemming=False).preprocess,
                               Preprocessor(strip_accents=True, twitter_symbols='normalize', stemming=True).preprocess),
        'vect__binary': (True, False),
        'vect__stop_words': (None, spanish_stopwords),
        'clf':(MultinomialNB(), KNeighborsClassifier(), DecisionTreeClassifier(), SVC())
    }
]
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=skf, verbose=1, scoring=scoring, refit='f1_micro')
    grid_search.fit(data, label)
    print("best_params:", grid_search.best_params_)
    print("best_score:", grid_search.best_score_)
    print(pd.DataFrame(grid_search.cv_results_).to_string())

