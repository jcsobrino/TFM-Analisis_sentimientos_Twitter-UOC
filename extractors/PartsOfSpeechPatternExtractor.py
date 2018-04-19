import re
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from util.Preprocessor import Preprocessor


class PartsOfSpeechPatternExtractor(BaseEstimator, TransformerMixin):

    POS_PATTERNS = ['na', 'nn', 'an', 'npn', 'vn', 'vpn']

    _vectorizer = None
    _processor = Preprocessor()

    def __init__(self, posHelper):
        self._pos_helper = posHelper
    
    def transform(self, data, y=None):
        result = []

        for tweet in data:
            result.append(self.pos_tag(tweet))

        if self._vectorizer == None :
            self._vectorizer = DictVectorizer(sparse=False)
            self._vectorizer.fit(result)

        return self._vectorizer.transform(result)

    def pos_tag(self, tweet):
        result = []
        tokens = self._tokenizer.tokenize(tweet)
        pos_tweet = self._pos_helper.pos_tag(tokens)
        words = [w for w,t in pos_tweet]
        pos_tags = ''.join([t for p,t in pos_tweet])

        for pattern in self._patters:
            x = [(m.start(0), m.end(0)) for m in re.finditer(pattern, pos_tags)]
            for i,j in x:
                result.append('_'.join(words[i:j]).lower())

        return Counter(result)

    def cosa(self, tweet):
        res = []
        pos = self.get_parts_of_speech(tweet)
        words = [w for w,t in pos]
        my_p = ''.join([p[1][0] for p in pos])

        for pattern in self._patters:
            x = [(m.start(0), m.end(0)) for m in re.finditer(pattern, my_p)]
            for i,j in x:
                res.append('_'.join(words[i:j]).lower())

        return Counter(res)

    def fit(self, df, y=None):
        return self