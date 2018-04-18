from collections import Counter

from nltk import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from util.PartsOfSpeechHelper import PartsOfSpeechHelper


class PartsOfSpeechExtractor(BaseEstimator, TransformerMixin):

    _vectorizer = None
    _tokenizer = TweetTokenizer()
    _pos_helper = PartsOfSpeechHelper()

    def __init__(self):
        pass

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            result.append(self.postag(tweet))

        if self._vectorizer == None :
            self._vectorizer = DictVectorizer(sparse=False)
            self._vectorizer.fit(result)

        return self._vectorizer.transform(result)

    def postag(self, tweet):
        pos_tweet = self._pos_helper.postag(tweet)
        return Counter([p[0] for p in pos_tweet])

    def fit(self, df, y=None):
        return self