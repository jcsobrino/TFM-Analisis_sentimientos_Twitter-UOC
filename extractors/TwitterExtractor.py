from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class TwitterExtractor(BaseEstimator, TransformerMixin):

    _normalizer = Preprocessor().normalize_tweet_elements

    def __init__(self):
        pass

    def transform(self, data, y=None):
        res = []

        for d in data:
            p = self._normalizer(d)
            res.append([p.count('_mention'),
                        p.count('_url'),
                        p.count('_hashtag')])

        return res

    def fit(self, df, y=None):
        return self