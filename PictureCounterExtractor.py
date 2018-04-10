import csv

import io
from sklearn.base import BaseEstimator, TransformerMixin


class PictureCounterExtractor(BaseEstimator, TransformerMixin):

    _pos = io.open('symbols/positive_symbols.csv', encoding='utf-8').read().splitlines()
    _neg = io.open('symbols/negative_symbols.csv', encoding='utf-8').read().splitlines()
    _neu = io.open('symbols/neutral_symbols.csv', encoding='utf-8').read().splitlines()

    def __init__(self):
        pass

    def transform(self, data, y=None):
        res= [[self.count_pict(tweet, self._pos),
                 self.count_pict(tweet, self._neg),
                 self.count_pict(tweet, self._neu),]
                for tweet in data]
        return res

    def count_pict(self, text, pict):
        return sum(True for c in pict if c in text)

    def fit(self, df, y=None):
        return self