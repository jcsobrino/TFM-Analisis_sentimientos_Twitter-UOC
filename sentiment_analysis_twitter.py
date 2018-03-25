import csv
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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
    return data[:500], label[:500]


def read_lexicon(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file]
    return lines


def preprocess_corpus_example(example):
    # text = re.sub(r'@\S+', '_user', text)
    # text = re.sub(r'#\S+', '_hashtag', text)
    # text = re.sub(r'https?:\S+', '_url', text)
    example = re.sub(r'(.)\1+', r'\1', example)
    return example
    # return re.sub(r'(@|#|https?:)\S+', '', text)


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

data, label = read_corpus("datasets/global_dataset.csv")

pipeline = Pipeline([('vect', CountVectorizer(strip_accents='unicode',
                                              analyzer='word',
                                              preprocessor=preprocess_corpus_example,
                                              lowercase=True)),
                     ('clf', LinearSVC()),
                     ])

parameters = {
    'vect__analyzer': ('word', stemmed_words),
    'vect__binary': (True, False),
    'vect__stop_words': (None, spanish_stopwords, [])
}

grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, cv=10)
grid_search.fit(data, label)
print("best_params:",grid_search.best_params_)
print("best_score:", grid_search.best_score_)
