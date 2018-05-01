from collections import Counter

from nltk import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from util.PartsOfSpeechHelper import PartsOfSpeechHelper
from util.Preprocessor import Preprocessor


class PartsOfSpeechPatternExtractor(BaseEstimator, TransformerMixin):

    POS_PATTERNS = [('NOUN','ADJ'), ('NOUN','NOUN'), ('ADJ','NOUN'), ('VERB','NOUN'), ('AUX','NOUN'),
                    ('NOUN','PRON','NOUN'), ('VERB','PRON','NOUN'), ('AUX','PRON','NOUN')]
    IGNORE_TAGS = ['PUNCT']

    _vectorizer = None
    _tokenizer = TweetTokenizer(reduce_len=True)
    _processor = Preprocessor(stemming=True)
    _pos_helper = PartsOfSpeechHelper()

    def __init__(self):
        pass
    
    def transform(self, data, y=None):
        result = []

        for tweet in data:
            result.append(self.get_patterns(tweet))

        if self._vectorizer == None :
            self._vectorizer = DictVectorizer(sparse=False)
            self._vectorizer.fit(result)

        return self._vectorizer.transform(result)

    def get_patterns(self, tweet):
        result = []
        tokens = self._tokenizer.tokenize(tweet)
        pos_tags = self._pos_helper.pos_tag(tokens)
        if len(pos_tags) > 1:
            pos_tags = [p for p in pos_tags if p[1] not in self.IGNORE_TAGS]
            words, tags = zip(*pos_tags)

            for pattern in self.POS_PATTERNS:
                found = self.find_sublist(list(pattern), list(tags))
                for i,j in found:
                    # Added patterns instead of tokens
                    result.append('_'.join(list(pattern)))
                    # result.append(self._processor.preprocess(' '.join(words[i:j])))

        return Counter(result)

    def fit(self, df, y=None):
        return self

    def find_sublist(self, sl, l):
        results = []
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                results.append((ind, ind + sll))

        return results