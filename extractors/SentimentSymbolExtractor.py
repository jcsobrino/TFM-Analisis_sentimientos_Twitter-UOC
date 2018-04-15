import io

from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class SentimentSymbolExtractor(BaseEstimator, TransformerMixin):

    _pos = io.open('sentiment-symbols/positive_symbols.txt', encoding='utf-8').read().splitlines()
    _neg = io.open('sentiment-symbols/negative_symbols.txt', encoding='utf-8').read().splitlines()
    _neu = io.open('sentiment-symbols/neutral_symbols.txt', encoding='utf-8').read().splitlines()
    _processor = Preprocessor(tweet_elements='remove')

    def __init__(self):
        pass

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = self._processor.preprocess(tweet)
            result.append([sum(tweet.count(e) for e in self._pos) + tweet.count('LAUGH'),
                           sum(tweet.count(e) for e in self._neg),
                           sum(tweet.count(e) for e in self._neu)])

        return result

    def fit(self, df, y=None):
        return self