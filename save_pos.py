import os
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from baseline_model import read_corpus

java_path = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
path_to_model = 'stanford_nlp/spanish.tagger'
path_to_jar = 'stanford_nlp/stanford-postagger-3.8.0.jar'

message_train, label_train = read_corpus("datasets/train_dataset_30.csv")
message_test, label_test = read_corpus("datasets/test_dataset_30.csv")

st = POS_Tag(model_filename=path_to_model, path_to_jar=path_to_jar)

def save_pos(data, filename):
    with open(filename, "a", encoding="utf-8") as afile:
        for d in data:
            afile.write(repr(st.tag([d])))
            afile.write('\n')

def read_pos(filename):
    data = []
    with open(filename,"r", encoding="utf-8") as f:
        for line in f:
            data.append(eval(line))
    return dict([(' '.join([h[0] for h in d]), d) for d in data])

save_pos(message_train[:1000], 'message_train_pos.txt')
save_pos(message_test[:1000], 'message_test_pos.txt')

read_pos('message_train_pos.txt')