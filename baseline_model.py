import csv
import time

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from Preprocessor import Preprocessor

spanish_stopwords = stopwords.words('spanish')

def read_corpus(filename):
    message = []
    label = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            message.append(row[1])
            label.append(row[2])
    return message, label

# corpus data
message, label = read_corpus("datasets/subset_dataset_30.csv")

scoring = {'accuracy': 'accuracy',
           'precision_macro': 'precision_macro',
           'recall_macro': 'recall_macro',
           'f1_macro': 'f1_macro',
           'precision_micro': 'precision_micro',
           'recall_micro': 'recall_micro',
           'f1_micro': 'f1_micro',
           'precision_weighted': 'precision_weighted',
           'recall_weighted': 'recall_weighted',
           'f1_weighted': 'f1_weighted'}

pipeline = Pipeline([('vect', None),
                    ('clf', None)])

weighted_binary = CountVectorizer(binary=True)
weighted_absolute = CountVectorizer(binary=False)
weighted_terms_frequency = TfidfVectorizer(use_idf=False)
weighted_tfidf = TfidfVectorizer(use_idf=True)

parameters = [
    {
        'vect': (weighted_binary,
                 weighted_absolute,
                 weighted_terms_frequency,
                 weighted_tfidf),
        'vect__preprocessor': (Preprocessor(twitter_symbols='remove', stemming=False).preprocess,
                               Preprocessor(twitter_symbols='remove', stemming=True).preprocess,
                               Preprocessor(twitter_symbols='normalized', stemming=False).preprocess,
                               Preprocessor(twitter_symbols='normalized', stemming=True).preprocess),
        'vect__stop_words': (None, spanish_stopwords),
        'clf':(MultinomialNB(), KNeighborsClassifier(), DecisionTreeClassifier(), LinearSVC())
    }
]

if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=skf, verbose=1, scoring=scoring, refit='f1_weighted')
    grid_search.fit(message, label)
    print("best_params:", grid_search.best_params_)
    print("best_score:", grid_search.best_score_)
    print(pd.DataFrame(grid_search.cv_results_).to_string())
    pd.DataFrame(grid_search.cv_results_).to_csv(path_or_buf=str(int(time.time()))+'.csv', quoting=csv.QUOTE_NONNUMERIC)

