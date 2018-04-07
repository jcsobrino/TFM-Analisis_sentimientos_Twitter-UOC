import csv

import pandas as pd
import time
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
    data = []
    label = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row[1])
            label.append(row[2])
    return data, label

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
        'vect': (CountVectorizer(binary=False),
                 CountVectorizer(binary=True),
                 TfidfVectorizer(use_idf=False),
                 TfidfVectorizer(use_idf=True)),
        'vect__preprocessor': (Preprocessor(stemming=False).preprocess,
                               Preprocessor(stemming=True).preprocess),
        'vect__stop_words': (None, spanish_stopwords),
        'clf':(MultinomialNB(), KNeighborsClassifier(), DecisionTreeClassifier(), LinearSVC())
    }
]
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=skf, verbose=1, scoring=scoring, refit='f1_micro')
    grid_search.fit(data, label)
    print("best_params:", grid_search.best_params_)
    print("best_score:", grid_search.best_score_)
    print(pd.DataFrame(grid_search.cv_results_).to_string())
    pd.DataFrame(grid_search.cv_results_).to_csv(path_or_buf=str(int(time.time()))+'.csv', quoting=csv.QUOTE_NONNUMERIC)

