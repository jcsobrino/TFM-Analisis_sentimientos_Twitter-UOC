from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from Preprocessor import Preprocessor


class TwitterExtractor(BaseEstimator, TransformerMixin):

    _processor = Preprocessor(twitter_symbols='normalize')

    def __init__(self):
        pass

    def transform(self, data, y=None):
        res = []

        for d in data:
            p = self._processor.preprocess(d)
            res.append([p.count('_mention'), p.count('_url'), p.count('_hashtag')])

        return res

    def fit(self, df, y=None):
        return self