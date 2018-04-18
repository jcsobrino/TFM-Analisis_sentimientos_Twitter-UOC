from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class LaughExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = Preprocessor.normalize_laughs(tweet)
            result.append([tweet.count(Preprocessor.LAUGH)])

        return preprocessing.normalize(result)

    def fit(self, df, y=None):
        return self