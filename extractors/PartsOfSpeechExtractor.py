import os
import re
from collections import Counter

from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

java_path = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

path_to_model = 'stanford-postagger/spanish.tagger'
path_to_jar = 'stanford-postagger/stanford-postagger-3.8.0.jar'

class PartsOfSpeechExtractor(BaseEstimator, TransformerMixin):

    _vectorizer = None
    _nlp = POS_Tag(model_filename=path_to_model, path_to_jar=path_to_jar)

    def __init__(self):
        self._cache = self._read_pos_cache('datasets/message_train_pos.txt')

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            result.append(self.get_parts_of_speech(tweet))

        if self._vectorizer == None :
            self._vectorizer = DictVectorizer(sparse=False)
            self._vectorizer.fit(result)

        return self._vectorizer.transform(result)

    def get_parts_of_speech(self, text):
        key = re.sub(' +',' ',text)
        key = re.sub('_', '', key)
        if key in self._cache:
            parts_of_speech = self._cache.get(key)
        else:
            print(text)
            parts_of_speech = self._nlp.tag([text])

        return Counter(list(zip(*parts_of_speech))[1])

    def fit(self, df, y=None):
        return self

    def _read_pos_cache(self, filename):
        data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data.append(eval(line))
        return dict([(' '.join([h[0] for h in d]), d) for d in data])
