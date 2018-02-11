import xml.etree.ElementTree as etree
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import numpy as np
import re
import unidecode
import nltk.tokenize
from collections import Counter
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost', port=9000, lang='es')
spanish_stopwords = stopwords.words('spanish')

def readData(xmlfile):
    tree = etree.parse(xmlfile)
    root = tree.getroot()
    data = []
    labels = []

    for tweet in root:
        text = tweet.find('content').text
        label = tweet.find('sentiments/polarity/value').text
        data.append(text)
        labels.append(label)

    return data, labels

def readLexicon(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file]
    return lines

def preprocessor(text):
    # text = re.sub(r'@\S+', '_user', text)
    # text = re.sub(r'#\S+', '_hashtag', text)
    # text = re.sub(r'https?:\S+', '_url', text)
    text = re.sub(r'(.)\1+', r'\1', text)
    return text
    #return re.sub(r'(@|#|https?:)\S+', '', text)

data, labels = readData("datasets/general-train-tagged-3l.xml")
data_test, labels_test = readData("datasets/general-test-tagged-3l.xml")

positive_words = Counter(readLexicon("iSOL/positivas_mejorada.csv"))
negative_words = Counter(readLexicon("iSOL/negativas_mejorada.csv"))

# text_clf = Pipeline([('vect', CountVectorizer(strip_accents='unicode',
#                                               analyzer ='word',
#                                               #preprocessor=preprocessor,
#                                               ngram_range=(1,3),
#                                               lowercase = True)),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', LinearSVC()),
# ])

def get_lexicon_score(text):
    result = []
    for t in text:
        t = t.lower()
        t = unidecode.unidecode(t)  # hay una mejora cuando se eliminan tildes
        #t = preprocessor(t)
        tokens = nltk.word_tokenize(t)
        pos = len(list((Counter(tokens) & positive_words).elements()))
        neg = len(list((Counter(tokens) & negative_words).elements()))
        #result.append((pos,neg))
        result.append([pos, neg])

    return result


def get_pos_data(text):
    result = []
    temp = []
    for t in text:
        temp.append(nlp.pos_tag(t))
    for t in temp:
        result.append(' '.join([y for x,y in t]))
    return result

text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('vect', CountVectorizer(strip_accents='unicode',
                                     #preprocessor=preprocessor,
                                              analyzer ='word',
                                              ngram_range=(1,3),
                                              lowercase = True)),
            ('tfidf', TfidfTransformer())
        ])),
        ('ngram_postag_tf_idf', Pipeline([
            ('pos', FunctionTransformer(get_pos_data, validate=False)),
            ('vect_postag', CountVectorizer(analyzer ='word',
                                            ngram_range=(1,3),
                                            lowercase= False)),
            ('tfidf_posttag', TfidfTransformer())
        ])),
        ('lexicon', FunctionTransformer(get_lexicon_score, validate=False))
    ])),
    ('clf', LinearSVC()),
])


text_clf.fit(data, labels)
predicted = text_clf.predict(data_test)

print(np.mean(predicted == labels_test))
print(classification_report(labels_test, predicted, list(set(labels))))
