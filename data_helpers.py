import re
import numpy as np
from sklearn.model_selection import train_test_split
import os
import collections
import argparse

parser = argparse.ArgumentParser(description='datasets_config')
parser.add_argument('--positive_path', default=r'./data/rt-polaritydata/rt-polarity.pos',
                    help='positive file')
parser.add_argument('--negative_path', default=r'./data/rt-polaritydata/rt-polarity.neg',
                    help='negative epoch')
args = parser.parse_args()
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " ", string)
    string = re.sub(r"n\'t", " ", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file, negative_data_file):

    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def preprocess():
    positive_data_file = args.positive_path
    negative_data_file = args.negative_path
    dev_sample_percentage = 0.1

    # load rt-polarity data
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file)

    # get word2index
    counter = collections.Counter()
    for text in x_text:
        text = text.split(' ')
        for word in text:
            counter[word] += 1
    word2index = collections.defaultdict(int)
    for wid, word in enumerate(counter.most_common()):
        word2index[word[0]] = wid+1
    word2index['PAD'] = 0

    # change word to index
    x_text_numeral = []
    for text in x_text:
        temp_text = []
        text = text.split(' ')
        for word in text:
            temp_text.append(word2index[word])
        x_text_numeral.append(temp_text)

    max_len = 0
    for i in range(len(x_text_numeral)):
        if len(x_text_numeral[i]) > max_len:
            max_len = len(x_text_numeral[i])

    for i in range(len(x_text_numeral)):
        if len(x_text_numeral[i]) < max_len:
            x_text_numeral[i] += (max_len - len(x_text_numeral[i])) * [word2index['PAD']]
    # 随机排列数据
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_text_numeral = np.array(x_text_numeral)
    x_shuffled = x_text_numeral[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    dev_sample_index = 0.8*len(shuffle_indices)
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x_text_numeral, y, x_shuffled, y_shuffled, x_text
    return x_train, y_train, x_dev, y_dev, word2index

def process_txt():
    content, label = load_data_and_labels(args.positive_path, args.negative_path)
    dev_sample_percentage = 0.1
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(label)))
    np.random.seed(1000)
    shuffle_indices = np.random.permutation(np.arange(len(label)))
    y_shuffled = label[shuffle_indices]
    x = np.array(content)
    x_shuffled = x[shuffle_indices]
    x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train, y_train, x_test, y_test


if __name__=="__main__":
    x_train, y_train, x_test, y_test = process_txt()
    train_txt = list()
    test_txt = list()
    for index, item in enumerate(x_train):
        label = str(y_train[index])
        temp_label = str(0) if label=='[0 1]' else str(1)
        temp = temp_label+'\t'+item
        train_txt.append(temp)
    for index, item in enumerate(x_test):
        label = str(y_test[index])
        temp_label = str(0) if label=='[0 1]' else str(1)
        temp = temp_label+'\t'+item
        test_txt.append(temp)
    with open('./data/train.txt', 'w', newline='') as f:
        for item in train_txt:
            f.write(item+'\n')
    with open('./data/test.txt', 'w', newline='') as f:
        for item in test_txt:
            f.write(item+'\n')

