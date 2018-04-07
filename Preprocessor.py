import re

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import strip_accents_unicode

class Preprocessor:

    _stemmer = SnowballStemmer('spanish')

    def __init__(self, strip_accents=False, twitter_symbols=None, stemming=False):
        self.strip_accents = strip_accents
        self.twitter_symbols = twitter_symbols
        self.stemming = stemming

    def preprocess(self, message):
        # convert to lowercase
        message = message.lower()
        # remove numbers, carriage returns and retweet old-style method
        message = re.sub(r'(\d+|\n|\brt\b)', '', message)

        if self.strip_accents:
            # remove accents
            message = strip_accents_unicode(message)

        if self.twitter_symbols == 'remove':
            # remove mentions, hashtags and urls
            message = re.sub(r'(@|#|https?:)\S+', '', message)
        elif self.twitter_symbols == 'normalize':
            # normalize mentions, hashtags and urls
            message = re.sub(r'@\S+', '_user_', message)
            message = re.sub(r'#\S+', '_hashtag_', message)
            message = re.sub(r'https?:\S+', '_url_', message)

        # remove repeated characters
        message = re.sub(r'([^clnr])\1+', r'\1', message)
        message = re.sub(r'([clnr])\1{3,}', r'\1\1', message)

        if self.stemming:
            message = ' '.join(self._stemmer.stem(w) for w in message.split())

        return message