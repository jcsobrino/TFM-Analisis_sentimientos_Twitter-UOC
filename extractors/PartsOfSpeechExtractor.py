from collections import Counter

from nltk import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from util.PartsOfSpeechHelper import PartsOfSpeechHelper


class PartsOfSpeechExtractor(BaseEstimator, TransformerMixin):

    IGNORE_TAGS = ['PUNCT', 'CCONJ']
    _vectorizer = None
    _tokenizer = TweetTokenizer(reduce_len=True)
    _pos_helper = PartsOfSpeechHelper()

    def __init__(self):
        pass

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
        return Counter([t for w,t in pos_tweet if t not in self.IGNORE_TAGS])

    def fit(self, df, y=None):
        return self