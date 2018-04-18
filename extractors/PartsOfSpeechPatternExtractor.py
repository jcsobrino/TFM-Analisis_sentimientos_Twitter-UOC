import os

import re
from collections import Counter
from nltk import TweetTokenizer
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from sklearn.feature_extraction import DictVectorizer

from util.PartsOfSpeechHelper import PartsOfSpeechHelper
from util.Preprocessor import Preprocessor

class PartsOfSpeechPatternExtractor(BaseEstimator, TransformerMixin):

    POS_PATTERNS = ['na', 'nn', 'an', 'npn', 'vn', 'vpn']

    _vectorizer = None
    _tokenizer = TweetTokenizer()
    _processor = Preprocessor()
    _pos_helper = PartsOfSpeechHelper()

    def __init__(self):
        pass
    
    def transform(self, data, y=None):
        result = []

        for tweet in data:
            #tweet = self._processor.preprocess(tweet)
            result.append(self.cosa(tweet))
            print(len(result))

        if self._vectorizer == None :
            self._vectorizer = DictVectorizer(sparse=False)
            self._vectorizer.fit(result)

        return self._vectorizer.transform(result)

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

    def get_parts_of_speech(self, text):
        key = self.generate_key(text)

        if key in self._cache:
            parts_of_speech = self._cache.get(key)
        else:
            parts_of_speech = self._nlp.tag([text])

        return parts_of_speech

    def fit(self, df, y=None):
        return self