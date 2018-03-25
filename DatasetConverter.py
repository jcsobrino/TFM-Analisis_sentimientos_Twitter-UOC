import xml.etree.ElementTree as etree
import csv

class DatasetConverter:

    @staticmethod
    def general_format_to_list(filename):
        tree = etree.parse(filename)
        root = tree.getroot()
        data = []

        for tweet in root:
            tweetId = tweet.find('tweetid').text
            content = tweet.find('content').text
            polarityValue = tweet.find('sentiments/polarity/value').text
            #polarityType = tweet.find('sentiments/polarity/type').text
            data.append([tweetId, content.replace('\n',' '), polarityValue])

        return data

    @staticmethod
    def politics_format_to_list(filename):
        tree = etree.parse(filename)
        root = tree.getroot()
        data = []

        for tweet in root:
            tweetId = tweet.find('tweetid').text
            content = tweet.find('content').text
            aux = next((e for e in tweet.findall('sentiments/polarity') if e.find('entity') == None), None)
            if aux != None:
                polarityValue = aux.find('value').text
                #polarityType = aux.find('type').text
                data.append([tweetId, content.replace('\n',' '), polarityValue])

        return data

    @staticmethod
    def intertass_format_to_list(filename, qrel=None):
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


qrel = DatasetConverter.gold_standard_to_dict("datasets/intertass-sentiment.qrel")

data = []
data.extend(DatasetConverter.general_format_to_list("datasets/general-test-tagged-3l.xml"))
data.extend(DatasetConverter.general_format_to_list("datasets/general-train-tagged-3l.xml"))
data.extend(DatasetConverter.intertass_format_to_list("datasets/intertass-development-tagged.xml"))
data.extend(DatasetConverter.intertass_format_to_list("datasets/intertass-test.xml", qrel))
data.extend(DatasetConverter.intertass_format_to_list("datasets/intertass-train-tagged.xml"))
data.extend(DatasetConverter.politics_format_to_list("datasets/politics-test-tagged.xml"))

DatasetConverter.list_to_csv(data, 'datasets/global_dataset.csv')