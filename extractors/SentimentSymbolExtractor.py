import io

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class SentimentSymbolExtractor(BaseEstimator, TransformerMixin):

    _preprocessor = Preprocessor(twitter_features=Preprocessor.REMOVE)

    def __init__(self):
        self._pos_symbols = self.file_to_list('sentiment-symbols/positive_symbols.txt')
        self._neg_symbols = self.file_to_list('sentiment-symbols/negative_symbols.txt')
        self._neu_symbols = self.file_to_list('sentiment-symbols/neutral_symbols.txt')

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = self._preprocessor.preprocess(tweet)
            result.append([sum(tweet.count(symbol) for symbol in self._pos_symbols),
                           sum(tweet.count(symbol) for symbol in self._neg_symbols),
                           sum(tweet.count(symbol) for symbol in self._neu_symbols)])

        return preprocessing.normalize(result)

    def fit(self, df, y=None):
        return self

    def file_to_list(self, filename):
        return io.open(filename, encoding='utf-8').read().splitlines()