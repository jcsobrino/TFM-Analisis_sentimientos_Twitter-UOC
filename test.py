import xml.etree.ElementTree as etree
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.metrics import recall, precision, f_measure, accuracy
import re
import unidecode
import collections
from nltk.classify import SklearnClassifier
from numpy import show_config
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

stopwords = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')


def createListOfData(xmlfile):
    tree = etree.parse(xmlfile)
    root = tree.getroot()
    data = []

    for tweet in root:
        text = tweet.find('content').text
        label = tweet.find('sentiments/polarity/value').text

        if label == 'P' or label == 'N':
             label = label
        else:
             label = 'NOSEN'
        data.append((text, label))

    return data

    counter = Counter(elem[1] for elem in data)

    cosa = []

    for type in counter.keys():
        aux = [(x,y) for (x,y) in data if y == type]
        aux = aux[:min(counter.values())]
        cosa.extend(aux)

    return cosa

def preprocessText(text):

    text = text.lower()
    tokens = text.split()
    tokens = [x for x in tokens if not x.startswith('@') and not x.startswith('#') and not x.startswith('http')]
    text = ' '.join(tokens)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = unidecode.unidecode(text) #hay una mejora cuando se eliminan tildes
    tokens = word_tokenize(text)
    tokens = [x for x in tokens if x[0].isalpha()] #and x not in stopwords]
    return tokens

def removeStopwords(tokens):
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

def extractFeatures(tokens):
    tokens = list(set(tokens))
    return Counter(tokens)
    #freq = nltk.FreqDist(tokens)
    #return dict([(term, freq.freq(term)) for term in freq.keys()])
    tokens = list(set(tokens))
    return dict([(term, True) for term in tokens])



training_data = createListOfData("datasets/general-train-tagged-3l.xml")
test_data = createListOfData("datasets/general-test-tagged-3l.xml")


training_data_featuresets = [(extractFeatures(preprocessText(n)), g) for (n, g) in training_data]
test_data_featuresets = [(extractFeatures(preprocessText(n)), g) for (n, g) in test_data]

#classifier = nltk.NaiveBayesClassifier.train(training_data_featuresets)
classifier = SklearnClassifier(LinearSVC())
classifier.train(training_data_featuresets+test_data_featuresets)



#print(nltk.classify.accuracy(classifier, test_data_featuresets))
#print(classifier.show_most_informative_features(10))

print(nltk.classify.accuracy(classifier, test_data_featuresets))

p = classifier.classify_many([x for x,y in test_data_featuresets])
v = [y for x,y in test_data_featuresets]

print(classification_report(v,p, classifier.labels()))

# refsets = collections.defaultdict(set)
# testsets = collections.defaultdict(set)
#
# for i, (feats, label) in enumerate(test_data_featuresets):
#     refsets[label].add(i)
#     observed = classifier.classify(feats)
#     testsets[observed].add(i)
#
# for label in refsets.keys():
#     print(label + ' precision:', precision(refsets[label], testsets[label]))
#     print(label + ' recall:', recall(refsets[label], testsets[label]))
#     print(label + ' F-measure:', f_measure(refsets[label], testsets[label]))
#     #print(label + ' Accuracy:', accuracy(refsets[label], testsets[label]))




#print(nltk.classify.accuracy(svmc, test_data_featuresets))
#print(svmc.show_most_informative_features(10))


#print(collections.Counter([b for(a, b) in test_data]))


