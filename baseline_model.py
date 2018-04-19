import csv

import pandas as pd
import time
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from util.DatasetHelper import DatasetHelper
from util.Preprocessor import Preprocessor

# corpus data
message, label = DatasetHelper.cvs_to_lists("datasets/train_dataset_30.csv")

# stop words
spanish_stopwords = stopwords.words('spanish')

# metrics
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

# pipeline
pipeline = Pipeline([('vectorizer', None),
                     ('classifier', None)])

# Tokenizer
tokenizer = TweetTokenizer().tokenize

# feature weights
bow_binary_term_ocurrences = CountVectorizer(binary=True, tokenizer=tokenizer)
bow_absolute_ocurrences = CountVectorizer(binary=False, tokenizer=tokenizer)
bow_term_frequency = TfidfVectorizer(use_idf=False, tokenizer=tokenizer)
bow_tfidf = TfidfVectorizer(use_idf=True, tokenizer=tokenizer)

parameters = [{
    'vectorizer': (bow_binary_term_ocurrences,
                   bow_absolute_ocurrences,
                   bow_term_frequency,
                   bow_tfidf),
    'vectorizer__preprocessor': (Preprocessor(twitter_features=Preprocessor.REMOVE).preprocess,
                                 Preprocessor(twitter_features=Preprocessor.REMOVE, stemming=True).preprocess,
                                 Preprocessor(twitter_features=Preprocessor.NORMALIZE).preprocess,
                                 Preprocessor(twitter_features=Preprocessor.NORMALIZE, stemming=True).preprocess),
    'vectorizer__stop_words': (None, spanish_stopwords),
    'classifier': (MultinomialNB(), LinearSVC(), DecisionTreeClassifier(), KNeighborsClassifier())
}]

if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=skf, verbose=5, scoring=scoring,
                               refit='f1_weighted', return_train_score=False)
    grid_search.fit(message, label)
    print("best_score:", grid_search.best_score_)
    pd.DataFrame(grid_search.cv_results_).to_csv(path_or_buf=str(int(time.time()))+'.csv', quoting=csv.QUOTE_NONNUMERIC)
