from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class TwitterExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = Preprocessor.process_twitter_features(tweet, tweet_elements=Preprocessor.NORMALIZE)
            result.append([tweet.count(Preprocessor.MENTION),
                           tweet.count(Preprocessor.URL),
                           tweet.count(Preprocessor.HASHTAG)])

        return result

    def fit(self, df, y=None):
        return self