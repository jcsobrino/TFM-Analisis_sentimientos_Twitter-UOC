import re

from nltk.stem import SnowballStemmer


class Preprocessor:

    _stemmer = SnowballStemmer('spanish')
    _diacritical_vowels = [('á','a'), ('é','e'), ('í','i'), ('ó','o'), ('ú','u'), ('ü','u')]
    _slang = [('d','de'), ('[qk]','que'), ('xo','pero'), ('xa', 'para'), ('[xp]q','porque'),('es[qk]', 'es que'),
              ('fvr','favor'),('(xfa|xf|pf|plis|pls|porfa)', 'por favor'), ('dnd','donde'), ('lol', '_laugh'), ('tb', 'también'),
              ('(tq|tk)', 'te quiero'), ('(tqm|tkm)', 'te quiero mucho'), ('x','por')]

    def __init__(self, twitter_symbols=None, stemming=False):
        self.twitter_symbols = twitter_symbols
        self.stemming = stemming

    def preprocess(self, message):
        # convert to lowercase
        message = message.lower()
        # remove numbers, carriage returns and retweet old-style method
        message = re.sub(r'(\d+|\n|\brt\b)', '', message)
        # remove vowels with diacritical marks
        for s,t in self._diacritical_vowels:
            message = re.sub(r'{0}'.format(s), t, message)
        # remove repeated characters
        message = re.sub(r'([^clnr])\1+', r'\1', message)
        message = re.sub(r'([clnr])\1{2,}', r'\1\1', message)
        # normalized laughs
        message = re.sub(r'\b(?=\w*[j])[aeiouj]{4,}\b', '_laugh', message)
        # normalized slang
        for s,t in self._slang:
            message = re.sub(r'\b{0}\b'.format(s), t, message)

        if self.twitter_symbols == 'remove':
            # remove mentions, hashtags and urls
            message = re.sub(r'(@|#|htt?ps?:)\S+', '', message)
        elif self.twitter_symbols == 'normalize':
            # normalize mentions, hashtags and urls
            message = re.sub(r'@\S+', '_mention', message)
            message = re.sub(r'#\S+', '_hashtag', message)
            message = re.sub(r'htt?ps?:\S+', '_url', message)

        if self.stemming:
            message = ' '.join(self._stemmer.stem(w) for w in message.split())

        return message

    def __repr__(self):
        return "Preprocessor([twitter_symbols={0}, stemming={1}])".format(self.twitter_symbols, self.stemming)

    def __str__(self):
        return "Preprocessor([twitter_symbols={0}, stemming={1}])".format(self.twitter_symbols, self.stemming)