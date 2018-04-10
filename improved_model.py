from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC

from PictureCounterExtractor import PictureCounterExtractor
from Preprocessor import Preprocessor
from baseline_model import read_corpus

message_train, label_train = read_corpus("datasets/train_dataset_30.csv")
message_test, label_test = read_corpus("datasets/test_dataset_30.csv")

preprocessor = Preprocessor(twitter_symbols='normalized', stemming=False).preprocess

pipeline = Pipeline([
    ('feats', FeatureUnion([
        ('vect', TfidfVectorizer(use_idf=False,
                                 preprocessor=preprocessor)),
        ('picture', PictureCounterExtractor())
    ])),
    ('clf', LinearSVC())
])

pipeline.fit(message_train, label_train)

y_prediction = pipeline.predict( message_test )

report = classification_report( label_test, y_prediction )

print(report)