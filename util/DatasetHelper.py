import xml.etree.ElementTree as etree
import csv

from sklearn.model_selection import train_test_split


class DatasetHelper:

    @staticmethod
    def general_tass_to_list(filename):
        tree = etree.parse(filename)
        root = tree.getroot()
        data = []

        for tweet in root:
            tweetId = tweet.find('tweetid').text
            content = tweet.find('content').text
            polarityValue = tweet.find('sentiments/polarity/value').text
            data.append([tweetId, content.replace('\n',' '), polarityValue])

        return data

    @staticmethod
    def politics_tass_to_list(filename):
        tree = etree.parse(filename)
        root = tree.getroot()
        data = []

        for tweet in root:
            tweetId = tweet.find('tweetid').text
            content = tweet.find('content').text
            aux = next((e for e in tweet.findall('sentiments/polarity') if e.find('entity') == None), None)
            if aux != None:
                polarityValue = aux.find('value').text
                data.append([tweetId, content.replace('\n',' '), polarityValue])

        return data

    @staticmethod
    def intertass_tass_to_list(filename, qrel=None):
        tree = etree.parse(filename)
        root = tree.getroot()
        data = []

        for tweet in root:
            tweetId = tweet.find('tweetid').text
            content = tweet.find('content').text
            polarityValue = tweet.find('sentiment/polarity/value').text
            if polarityValue == None:
                polarityValue = qrel[tweetId]

            data.append([tweetId, content.replace('\n',' '), polarityValue])

        return data

    @staticmethod
    def gold_standard_to_dict(filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            data = {rows[0]: rows[1] for rows in reader}

        return data

    @staticmethod
    def list_to_csv(data, filename):
        with open(filename, 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(data)

    @staticmethod
    def generate_train_test_subsets(data, size):
        codes = [d[0] for d in data]
        labels = [d[2] for d in data]
        codes_train, codes_test, labels_train, labels_test = train_test_split(codes, labels, train_size=size)
        train_data = [d for d in data if d[0] in codes_train]
        test_data = [d for d in data if d[0] in codes_test]
        return train_data, test_data

    @staticmethod
    def cvs_to_lists(filename):
        messages = []
        labels = []
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                messages.append(row[1])
                labels.append(row[2])
        return messages, labels

#qrel = DatasetHelper.gold_standard_to_dict("datasets/intertass-sentiment.qrel")

test = []
train = []
test.extend(DatasetHelper.general_tass_to_list("../datasets/tass/general-test-tagged-3l.xml"))
train.extend(DatasetHelper.general_tass_to_list("../datasets/tass/general-train-tagged-3l.xml"))
# data.extend(DatasetHelper.intertass_tass_to_list("datasets/intertass-development-tagged.xml"))
# data.extend(DatasetHelper.intertass_tass_to_list("datasets/intertass-test.xml", qrel))
# data.extend(DatasetHelper.intertass_tass_to_list("datasets/intertass-train-tagged.xml"))
# data.extend(DatasetHelper.politics_tass_to_list("datasets/politics-test-tagged.xml"))

# train, test = DatasetHelper.generate_train_test_subsets(data, size=0.3)

#DatasetHelper.list_to_csv(data, 'datasets/global_dataset.csv')
DatasetHelper.list_to_csv(train, '../datasets/standard_train_dataset.csv')
DatasetHelper.list_to_csv(test, '../datasets/standard_test_dataset.csv')
