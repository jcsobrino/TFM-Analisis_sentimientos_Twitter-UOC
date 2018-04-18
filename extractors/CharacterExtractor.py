import re

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class CharacterExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = Preprocessor.process_tweet_elements(tweet, tweet_elements="remove")
            result.append([
                tweet.count('!'),
                int(bool(re.search(r"(\w)\1{2,}", tweet))),
                len(re.findall(r'[A-Z]', tweet))/len(tweet) if len(tweet) > 0 else 0
            ])

        return preprocessing.normalize(result)

    def fit(self, df, y=None):
        return self