import os
from collections import Counter

import re
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from save_pos import read_pos

java_path = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

path_to_model = 'stanford_nlp/spanish.tagger'
path_to_jar = 'stanford_nlp/stanford-postagger-3.8.0.jar'

class POSExtractor(BaseEstimator, TransformerMixin):

    _vector = None
    _nlp = POS_Tag(model_filename=path_to_model, path_to_jar=path_to_jar)
    _cache = read_pos('datasets/message_train_pos.txt')

    def __init__(self):
        pass

    def transform(self, data, y=None):
        pos = []

        for d in data:
            pos.append(self.get_pos(d))

        if self._vector == None:
            self._vector = DictVectorizer(sparse=False)
            self._vector.fit(pos)

        return self._vector.transform(pos)

    def get_pos(self, text):
        key = re.sub(' +',' ',text)
        key = re.sub('_', '', key)
        if key in self._cache:
            pos = self._cache.get(key)
        else:
            print(text)
            pos = self._nlp.tag([text])

        return Counter([h[1] for h in pos])

    def fit(self, df, y=None):
        return self

