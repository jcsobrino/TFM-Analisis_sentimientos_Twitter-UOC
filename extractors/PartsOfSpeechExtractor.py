from collections import Counter

from nltk import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class PartsOfSpeechExtractor(BaseEstimator, TransformerMixin):

    _vectorizer = None
    _tokenizer = TweetTokenizer(reduce_len=True)

    def __init__(self, pos_helper):
        self._pos_helper = pos_helper

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            result.append(self.pos_tag(tweet))

        if self._vectorizer == None :
            self._vectorizer = DictVectorizer(sparse=False)
            self._vectorizer.fit(result)

        return self._vectorizer.transform(result)

    def pos_tag(self, tweet):
        tokens = self._tokenizer.tokenize(tweet)
        pos_tweet = self._pos_helper.pos_tag(tokens)
        return Counter([pos[1] for pos in pos_tweet])

    def fit(self, df, y=None):
        return self