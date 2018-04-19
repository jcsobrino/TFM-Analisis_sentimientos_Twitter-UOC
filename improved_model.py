from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC

from baseline_model import read_corpus
from extractors.CharacterExtractor import CharacterExtractor
from extractors.LaughExtractor import LaughExtractor
from extractors.LexiconExtractor import LexiconExtractor
from extractors.PartsOfSpeechExtractor import PartsOfSpeechExtractor
from extractors.PartsOfSpeechPatternExtractor import PartsOfSpeechPatternExtractor
from extractors.SentimentSymbolExtractor import SentimentSymbolExtractor
from extractors.TwitterExtractor import TwitterExtractor
from util.Preprocessor import Preprocessor

message_train, label_train = read_corpus("datasets/train_dataset_30.csv")
message_test, label_test = read_corpus("datasets/test_dataset_30.csv")

preprocessor = Preprocessor(twitter_features='normalize', stemming=False).preprocess

pipeline = Pipeline([
    ('feats', FeatureUnion([
         ('vect', TfidfVectorizer(use_idf=False,
                                  preprocessor=preprocessor)),
        ('sentiment_symbol', SentimentSymbolExtractor()),
        ('parts_of_speech', PartsOfSpeechPatternExtractor()),
        ('lexicon', LexiconExtractor())
        # ('laugh', LaughExtractor())
        # ('character', CharacterExtractor())
        # ('twitter', TwitterExtractor())
    ])),
    ('clf', LinearSVC())
])


l = 999999

pipeline.fit(message_train[:l], label_train[:l])

y_prediction = pipeline.predict( message_test[:l] )

report = classification_report( label_test[:l], y_prediction )

print(report)
print(accuracy_score(label_test[:l], y_prediction))
