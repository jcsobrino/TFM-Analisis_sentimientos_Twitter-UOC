import io
import re
import string

from nltk import TweetTokenizer
from nltk.util import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

from sklearn.feature_extraction import DictVectorizer

from util.Preprocessor import Preprocessor

class LexiconExtractor(BaseEstimator, TransformerMixin):

    _vector = DictVectorizer(sparse=False)
    _ngram_length = 5
    _reverse = ['no', 'ni', 'apenas', 'nada', 'tampoco', 'nunca', 'ningun', 'ninguna', 'ninguno', 'nadie', 'jamas']
    _tokenizer = TweetTokenizer()
    _neg = io.open('lexicon/negative_words.csv').read().splitlines()
    _pos = io.open('lexicon/positive_words.csv').read().splitlines()
    _processor = Preprocessor(tweet_elements='remove')

    def __init__(self):
        pass

    def transform(self, data, y=None):
        res = [self.detect(tweet) for tweet in data]
        return self._vector.fit_transform(res)

    def detect(self, text):
        res = []
        aux = self._processor.preprocess(text)
        aux = re.sub('[' + string.punctuation + ']', '', aux)
        aux = list(ngrams(self._tokenizer.tokenize(aux), self._ngram_length, pad_left=True))

        for ngram in aux:
            pre_words = ngram[:self._ngram_length-1]
            word = ngram[self._ngram_length-1]

            if word in self._pos:
                if any(w in pre_words for w in self._reverse):
                    res.append("NEG")
                else:
                    res.append("POS")

            elif word in self._neg:
                if any(w in pre_words for w in self._reverse):
                    res.append("POS")
                else:
                    res.append("NEG")

        return Counter(res)

    def fit(self, df, y=None):
        return self