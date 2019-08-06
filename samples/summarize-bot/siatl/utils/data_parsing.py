import glob
import html
import os

import pickle
from sklearn.model_selection import train_test_split

from sys_config import DATA_DIR


def clean_text(text):
    """
    Remove extra quotes from text files and html entities
    Args:
        text (str): a string of text

    Returns: (str): the "cleaned" text

    """
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text


def parse_file(file):
    """
    Read a file and return a dictionary of the data, in the format:
    tweet_id:{sentiment, text}
    """

    data = {}
    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        columns = line.rstrip().split("\t")
        tweet_id = columns[0]
        sentiment = columns[1]
        text = columns[2:]
        text = clean_text(" ".join(text))
        data[tweet_id] = (sentiment, text)
    return data


def load_data_from_dir(path):
    FILE_PATH = os.path.dirname(__file__)
    files_path = os.path.join(FILE_PATH, path)

    files = glob.glob(files_path + "/**/*.tsv", recursive=True)
    files.extend(glob.glob(files_path + "/**/*.txt", recursive=True))

    data = {}  # use dict, in order to avoid having duplicate tweets (same id)
    for file in files:
        file_data = parse_file(file)
        data.update(file_data)
    return list(data.values())


def load_scv2_dataset(test=False):
    path = os.path.join(DATA_DIR, 'SCv2-GEN', 'raw_forum.pickle')
    if test:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        test = [(data['texts'][x], data['info'][x]["label"], data['post_ids'][x]) for x in
                 data['test_ind']]
        X_test = [x[0] for x in test]
        y_test = [x[1] for x in test]
        pids_test = [x[2] for x in test]
        dataset = [X_test, y_test, pids_test]
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        train = [(data['texts'][x], data['info'][x]["label"]) for x in
                 data['train_ind']]
        val = [(data['texts'][x], data['info'][x]["label"]) for x in
               data['val_ind']]

        X_train = [x[0] for x in train]
        X_val = [x[0] for x in val]

        y_train = [x[1] for x in train]
        y_val = [x[1] for x in val]
        dataset = [X_train, y_train, X_val, y_val]
    return dataset


def load_forum_dataset(test=False):
    path = os.path.join(DATA_DIR, 'Forum', 'raw_forum.pickle')
    if test:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        test = [(data['texts'][x], data['info'][x]["label"], data['post_ids'][x], data['posts'][x]) for x in
                 data['test_ind']]
        X_test = [x[0] for x in test]
        y_test = [x[1] for x in test]
        pids_test = [x[2] for x in test]
        posts = [x[3] for x in test]
        dataset = [X_test, y_test, posts, pids_test, None]
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        train = [(data['texts'][x], data['info'][x]["label"], data['posts'][x]) for x in data['train_ind']]
        val = [(data['texts'][x], data['info'][x]["label"], data['posts'][x]) for x in data['val_ind']]

        X_train = [x[0] for x in train]
        X_val = [x[0] for x in val]

        y_train = [x[1] for x in train]
        y_val = [x[1] for x in val]

        post_train =[x[2] for x in train]
        post_val = [x[2] for x in val]

        dataset = [X_train, y_train, post_train, X_val, y_val, post_val]
    return dataset


def load_cnn_dataset(test=False):
    path = os.path.join(DATA_DIR, 'CNN', 'raw_cnn.pickle')
    if test:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        test = [(data['texts'][x], data['info'][x]["label"], data['post_ids'][x], data['human_summaries'][x], data['posts'][x]) for x in data['test_ind']]
        X_test = [x[0] for x in test]
        y_test = [x[1] for x in test]
        pids_test = [x[2] for x in test]
        human_summaries = [x[3] for x in test]
        posts = [x[4] for x in test]
        dataset = [X_test, y_test, posts, pids_test, human_summaries]
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        train = [(data['texts'][x], data['info'][x]["label"], data['posts'][x]) for x in data['train_ind']]
        val = [(data['texts'][x], data['info'][x]["label"], data['posts'][x]) for x in data['val_ind']]

        X_train = [x[0] for x in train]
        X_val = [x[0] for x in val]

        y_train = [x[1] for x in train]
        y_val = [x[1] for x in val]

        post_train =[x[2] for x in train]
        post_val = [x[2] for x in val]

        dataset = [X_train, y_train, post_train, X_val, y_val, post_val]
    return dataset

def load_sent17_dataset():
    train = load_data_from_dir(os.path.join(DATA_DIR,
                                            'sentiment2017/train'))
    X = [obs[1] for obs in train]
    y = [obs[0] for obs in train]
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.1,
                                                      stratify=y)
    dataset = [X_train, y_train, X_val, y_val]
    return dataset


def load_dataset(config, test=False):
    if config["data"]["dataset"] == "scv2":
        print('Loading SCV2 dataset')
        dataset = load_scv2_dataset(test)
    elif config["data"]["dataset"] == "cnn":
        print('Loading cnn dataset')
        dataset = load_cnn_dataset(test)
    elif config["data"]["dataset"] == "forum":
        print('Loading forum dataset')
        dataset = load_forum_dataset(test)
    else:
        print('Data set not found')
        # dataset = load_sent17_dataset()
    return dataset


def get_texts(path, classes):
    texts, labels = [], []
    for idx, label in enumerate(classes):
        files = glob.glob(path + "/" + label + "/" + "*.txt")
        for file in files:
            texts.append(open(file, "r", encoding="utf-8").read())
            labels.append(label)
    return texts, labels


def parse_affect(data_file):
    """

    Returns:
        X: a list of tweets
        y: a list of lists corresponding to the emotion labels of the tweets

    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()][1:]
    X = [d[1] for d in data]
    # dict.values() does not guarantee the order of the elements
    # so we should avoid using a dict for the labels
    y = [[int(l) for l in d[2:]] for d in data]

    return X, y