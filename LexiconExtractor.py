import io

from nltk import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin

class LexiconExtractor(BaseEstimator, TransformerMixin):

    _neg = io.open('iSOL/negativas_mejorada.csv', encoding='utf-8').read().splitlines()
    _pos = io.open('iSOL/positivas_mejorada.csv', encoding='utf-8').read().splitlines()

    def __init__(self):
        pass

    def transform(self, data, y=None):
        res= [[self.count(tweet, self._pos),
               self.count(tweet, self._neg)]
              for tweet in data]
        return res

    def count(self, text, words):
        return sum(True for word in TweetTokenizer(text) if word in words)

    def fit(self, df, y=None):
        return self