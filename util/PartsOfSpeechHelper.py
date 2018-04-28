import hashlib
import os
import pickle

from nltk.tag.stanford import StanfordPOSTagger

os.environ['JAVAHOME'] = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"

class PartsOfSpeechHelper:

    PATH_TO_MODEL = 'stanford-postagger/spanish-ud.tagger'
    PATH_TO_JAR = 'stanford-postagger/stanford-postagger-3.8.0.jar'

    _postag = StanfordPOSTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR)

    def __init__(self):
        self.load_file_into_cache('stanford-postagger/precalculated_ud_postags.pkl')

    def pos_tag(self, tokens):
        key = self.key(tokens)

        if key in self._cache:
            return self._cache.get(key)

        parts_of_speech = self._postag.tag(tokens)
        self._cache[key] = parts_of_speech

        return parts_of_speech

    def key(self, tokens):
        return hashlib.md5(''.join(tokens).encode('utf-8')).hexdigest()

    def load_file_into_cache(self, filename):
        with open(filename, 'rb') as f:
            self._cache = pickle.load(f)

    def save_cache_into_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._cache, f, pickle.HIGHEST_PROTOCOL)

    def _pos_tag_batch(self, tokens_list):
        parts_of_speech = self._postag.tag_sents(tokens_list)
        for index, item in enumerate(parts_of_speech):
            key = self.key(tokens_list[index])
            self._cache[key] = item
