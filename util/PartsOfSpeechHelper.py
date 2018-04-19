import os
from nltk import TweetTokenizer
from nltk.tag.stanford import StanfordPOSTagger
import hashlib

os.environ['JAVAHOME'] = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"

class PartsOfSpeechHelper:

    PATH_TO_MODEL = 'stanford-postagger/spanish.tagger'
    PATH_TO_JAR = 'stanford-postagger/stanford-postagger-3.8.0.jar'

    _tokenizer = TweetTokenizer()
    _postag = StanfordPOSTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR)

    def __init__(self):
        self._cache = self.load_precalculate_postags('stanford-postagger/precalculated_corpus_postags.txt')

    def postag(self, tokens):
        key = self.key(tokens)

        if key in self._cache:
            return self._cache.get(key)

        parts_of_speech = self._postag.tag(tokens)
        self._cache[key] = parts_of_speech

        return parts_of_speech

    def load_precalculate_postags(self, filename):
        data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data.append(eval(line))
        return dict([(self.key([w for w,t in pos_tags]), pos_tags) for pos_tags in data])

    def key(self, tokens):
        return hashlib.md5(''.join(tokens).encode('utf-8')).hexdigest()

    def save_cache_in_file(self, filename):
        with open(filename, "a", encoding="utf-8") as file:
            for k,d in self._cache:
                file.write(repr(k))
                file.write('\n')