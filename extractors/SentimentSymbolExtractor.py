import io

from sklearn.base import BaseEstimator, TransformerMixin


class SentimentSymbolExtractor(BaseEstimator, TransformerMixin):

    _pos = io.open('sentiment-symbols/positive_symbols.txt', encoding='utf-8').read().splitlines()
    _neg = io.open('sentiment-symbols/negative_symbols.txt', encoding='utf-8').read().splitlines()
    _neu = io.open('sentiment-symbols/neutral_symbols.txt', encoding='utf-8').read().splitlines()

    def __init__(self):
        pass

    def transform(self, data, y=None):
        res= [[sum(tweet.count(e) for e in self._pos),
               sum(tweet.count(e) for e in self._neg),
               sum(tweet.count(e) for e in self._neu)]
              for tweet in data]
        return res

    def fit(self, df, y=None):
        return self