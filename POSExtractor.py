import os
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag

java_path = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

home = 'D:/UOC/TFM/'
_path_to_model = home + '/stanford-postagger/models/spanish.tagger'
_path_to_jar = home + '/stanford-corenlp-full-2018-01-31/stanford-postagger.jar'

class POSExtractor(BaseEstimator, TransformerMixin):

    _st = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)

    def __init__(self):
        pass

    def transform(self, data, y=None):
        pos = []
        for d in data:
            pos.append(self.get_pos(d))
        return pos

    def get_pos(self, text):
        return Counter([h[1] for h in self._st.tag(text.split())])

    def fit(self, df, y=None):
        return self

