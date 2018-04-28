import io

from nltk import TweetTokenizer
from nltk.util import ngrams
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from util.Preprocessor import Preprocessor


class LexiconExtractor(BaseEstimator, TransformerMixin):

    NGRAM_LENGTH = 3
    REVERSE_WORDS = ['no', 'ni', 'tampoc', 'ningun']

    _tokenizer = TweetTokenizer()
    _preprocessor = Preprocessor(twitter_features=Preprocessor.REMOVE, stemming=True)

    def __init__(self):
        self._neg_words = self.file_to_list('lexicon/negative_words.txt')
        self._pos_words = self.file_to_list('lexicon/positive_words.txt')

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = self._preprocessor.preprocess(tweet)
            result.append(self.count_polarity_words(tweet))

        return preprocessing.normalize(result)

    def count_polarity_words(self, text):
        num_pos_words = 0
        num_neg_words = 0

        list_ngrams = list(ngrams(self._tokenizer.tokenize(text), self.NGRAM_LENGTH, pad_left=True))

        for ngram in list_ngrams:
            pre_words = ngram[:self.NGRAM_LENGTH-1]
            word = ngram[self.NGRAM_LENGTH-1]

            if word in self._pos_words:
                if any(w in pre_words for w in self.REVERSE_WORDS):
                    num_neg_words += 1
                else:
                    num_pos_words += 1

            elif word in self._neg_words:
                if any(w in pre_words for w in self.REVERSE_WORDS):
                    num_pos_words += 1
                else:
                    num_neg_words += 1

        return [num_pos_words, num_neg_words]

    def fit(self, df, y=None):
        return self

    def file_to_list(self, filename):
        return io.open(filename).read().splitlines()