from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from extractors.CharacterExtractor import CharacterExtractor
from extractors.LaughExtractor import LaughExtractor
from extractors.LexiconExtractor import LexiconExtractor
from extractors.PartsOfSpeechExtractor import PartsOfSpeechExtractor
from extractors.PartsOfSpeechPatternExtractor import PartsOfSpeechPatternExtractor
from extractors.SentimentSymbolExtractor import SentimentSymbolExtractor
from extractors.TwitterExtractor import TwitterExtractor
from util.DatasetHelper import DatasetHelper
from util.PartsOfSpeechHelper import PartsOfSpeechHelper
from util.Preprocessor import Preprocessor

# global corpus data
message_train, label_train = DatasetHelper.cvs_to_lists("datasets/train_dataset_30.csv")
message_test, label_test = DatasetHelper.cvs_to_lists("datasets/test_dataset_30.csv")

# corpus standard TASS 2012
#message_train, label_train = DatasetHelper.cvs_to_lists("datasets/standard_train_dataset.csv")
#message_test, label_test = DatasetHelper.cvs_to_lists("datasets/standard_test_dataset.csv")

# Tokenizer
tokenizer = TweetTokenizer().tokenize

# Preprocessor with stemming enabled
preprocessor = Preprocessor(twitter_features=None, stemming=True).preprocess

# Term Frequency
bow_term_frequency = TfidfVectorizer(use_idf=False, tokenizer=tokenizer, preprocessor=preprocessor)

pipeline = Pipeline([
    ('feats', FeatureUnion([
         ('vectorizer', bow_term_frequency),
         ('sentiment_symbol', SentimentSymbolExtractor()),
         ('parts_of_speech', PartsOfSpeechExtractor()),
         #('parts_of_speech_pattern', PartsOfSpeechPatternExtractor()),
         ('lexicon', LexiconExtractor()),
         ('laugh', LaughExtractor()),
         ('character', CharacterExtractor()),
         ('twitter', TwitterExtractor())
    ])),
    ('classifier', LinearSVC(C=0.5))
])


pipeline.fit(message_train, label_train)

y_prediction = pipeline.predict( message_test )

report = classification_report(label_test, y_prediction, digits=4)

print(report)
print(accuracy_score(label_test, y_prediction))
