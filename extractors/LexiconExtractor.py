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

    _vectorizer = None
    _ngram_length = 3
    _reverse = ['no', 'ni', 'tampoc', 'ningun']
    _tokenizer = TweetTokenizer()
    _neg = io.open('lexicon/negative_words.txt').read().splitlines()
    _pos = io.open('lexicon/positive_words.txt').read().splitlines()
    _processor = Preprocessor(tweet_elements='remove', stemming=True)

    def __init__(self):
        pass

    def transform(self, data, y=None):
        result = [self.detect(tweet) for tweet in data]
        if self._vectorizer == None :
            self._vectorizer = DictVectorizer(sparse=False)
            self._vectorizer.fit(result)

        return self._vectorizer.transform(result)

    def detect(self, text):
        res = []
        aux = self._processor.preprocess(text)
        aux = re.sub('[' + string.punctuation + ']', '', aux)
        aux = list(ngrams(self._tokenizer.tokenize(aux), self._ngram_length, pad_left=True))

        for ngram in aux:
            pre_words = ngram[:self._ngram_length-1]
            word = ngram[self._ngram_length-1]
            #word = self._processor.stem(word)

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