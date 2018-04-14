import io
import re
import string

from nltk import TweetTokenizer
from nltk.util import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

from sklearn.feature_extraction import DictVectorizer

from Preprocessor import Preprocessor

class LexiconExtractor(BaseEstimator, TransformerMixin):

    _vector = DictVectorizer(sparse=False)
    _ngram_length = 5
    _reverse = ['no', 'ni', 'apenas', 'nada', 'tampoco', 'nunca', 'ningun', 'ninguna', 'ninguno', 'nadie', 'jamas']
    _tokenizer = TweetTokenizer()
    _neg = io.open('iSOL/negativas_mejorada.csv').read().splitlines()
    _pos = io.open('iSOL/positivas_mejorada.csv').read().splitlines()
    _processor = Preprocessor(twitter_symbols='remove')

    def __init__(self):
        #self._vector.fit(['POS', 'NEG'])
        pass

    def transform(self, data, y=None):
        res = [self.detect(tweet) for tweet in data]
        return self._vector.fit_transform(res)

    def count(self, text, words):
        aux = self._processor.preprocess(text)
        return sum(True for word in self._tokenizer.tokenize(aux) if word in words)

    def detect(self, text):
        res = []
        aux = self._processor.preprocess(text)
        aux = re.sub('[' + string.punctuation + ']', '', aux)
        aux = list(ngrams(self._tokenizer.tokenize(aux), self._ngram_length, pad_left=True))

        for gram in aux:
            pre = gram[:self._ngram_length-1]
            word = gram[self._ngram_length-1]

            if self.is_word_in(word, self._pos):
                if self.is_any_word_reverse(pre):
                    res.append("NEG")
                else:
                    res.append("POS")

            if self.is_word_in(word, self._neg):
                if self.is_any_word_reverse(pre):
                    res.append("POS")
                else:
                    res.append("NEG")

        return Counter(res)

    def is_any_word_reverse(self, words):
        return any(i in words for i in self._reverse)

    def is_word_in(self, word, data):
        return word in data

    def fit(self, df, y=None):
        return self