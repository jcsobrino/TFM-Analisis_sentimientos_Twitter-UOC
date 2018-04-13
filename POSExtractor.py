import os
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from sklearn.feature_extraction import DictVectorizer
from stanfordcorenlp import StanfordCoreNLP


java_path = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

home = ''
path_to_model = home + 'stanford_nlp/spanish.tagger'
path_to_jar = home + 'stanford_nlp/stanford-postagger-3.8.0.jar'

port = 9000
host = 'http://localhost'

class POSExtractor(BaseEstimator, TransformerMixin):

    _vector = None
    #_nlp = StanfordCoreNLP(host, port=port, timeout=30000, lang='es')
    _nlp = POS_Tag(model_filename=path_to_model, path_to_jar=path_to_jar)

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
        return Counter([h[1] for h in self._nlp.tag([text])])

    def fit(self, df, y=None):
        return self

