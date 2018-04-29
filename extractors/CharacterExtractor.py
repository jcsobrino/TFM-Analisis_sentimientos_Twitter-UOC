import re
from itertools import groupby

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class CharacterExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = Preprocessor.process_twitter_features(tweet, twitter_features=Preprocessor.REMOVE)
            result.append([
                tweet.count('!'),
                self._max_consecutive_equals_characteres(tweet),
                len(re.findall(r'[A-Z]', tweet))
            ])

        return preprocessing.normalize(result)

    def fit(self, df, y=None):
        return self

    def _max_consecutive_equals_characteres(self, text):
        if len(text) == 0:
            return 0

        groups = groupby(text)
        return max(num for char, num in [(char, sum(1 for _ in group)) for char, group in groups])