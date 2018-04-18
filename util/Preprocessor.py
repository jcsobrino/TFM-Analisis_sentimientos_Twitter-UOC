import re

from nltk import word_tokenize
from nltk.stem import SnowballStemmer


class Preprocessor:

    MENTION = 'TWMENTION'
    HASHTAG = 'TWHASHTAG'
    URL = 'TWURL'
    LAUGH = 'TWLAUGH'

    DIACRITICAL_VOWELS = [('á','a'), ('é','e'), ('í','i'), ('ó','o'), ('ú','u'), ('ü','u')]
    SLANG = [('d','de'), ('[qk]','que'), ('xo','pero'), ('xa', 'para'), ('[xp]q','porque'),('es[qk]', 'es que'),
              ('fvr','favor'),('(xfa|xf|pf|plis|pls|porfa)', 'por favor'), ('dnd','donde'), ('lol', '_laugh'), ('tb', 'también'),
              ('(tq|tk)', 'te quiero'), ('(tqm|tkm)', 'te quiero mucho'), ('x','por')]

    _stemmer = SnowballStemmer('spanish')

    def __init__(self, tweet_elements=None, stemming=False):
        self._tweet_elements = tweet_elements
        self._stemming = stemming

    def preprocess(self, message):
        # convert to lowercase
        message = message.lower()
        # remove numbers, carriage returns and retweet old-style method
        message = re.sub(r'(\d+|\n|\brt\b)', '', message)
        # remove vowels with diacritical marks
        for s,t in self.DIACRITICAL_VOWELS:
            message = re.sub(r'{0}'.format(s), t, message)
        # remove repeated characters
        message = re.sub(r'(.)\1{2,}', r'\1\1', message)
        # normalized laughs
        message = self.normalize_laughs(message)
        # remove slang
        for s,t in self.SLANG:
            message = re.sub(r'\b{0}\b'.format(s), t, message)

        message = self.process_tweet_elements(message, self._tweet_elements)

        if self._stemming:
            message = ' '.join(self._stemmer.stem(w) for w in word_tokenize(message))

        return message

    @staticmethod
    def process_tweet_elements(self, message, tweet_elements):
        if tweet_elements == 'remove':
            # remove mentions, hashtags and urls
            message = re.sub(r'(@|#|https?:)\S+', '', message)
        elif tweet_elements == 'normalize':
            # normalize mentions, hashtags and urls
            message = re.sub(r'@\S+', self.MENTION, message)
            message = re.sub(r'#\S+', self.HASHTAG, message)
            message = re.sub(r'https?:\S+', self.URL, message)

        return message

    def stem(self, word):
        return self._stemmer.stem(word)

    @staticmethod
    def normalize_laughs(self, message):
        return re.sub(r'\b(?=\w*[j])[aeiouj]{4,}\b', self.LAUGH, message.lower())

    def __repr__(self):
        return "Preprocessor([tweet_elements={0}, stemming={1}])".format(self._tweet_elements, self._stemming)

    def __str__(self):
        return "Preprocessor([tweet_elements={0}, stemming={1}])".format(self._tweet_elements, self._stemming)