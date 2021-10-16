"""
使用sklearn-learn的SGDClassifier的partial_fit函数，从本地直接获取流式文件，并用小批次文档 训练 逻辑回归
"""

import os
import sys
import tarfile
import time
import pyprind
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import gzip
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emotions).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    """
    每次读入并返回一个文档, 测试：next(stream_docs(path='movie_data.csv'))
    """
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    """ 调用stream_docs读入文档流并通过参数size返回指定数量的文档
    """
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)

    except StopIteration:
        return None, None
    return docs, y


if __name__ == "__main__":
    # The `stop` is defined as earlier in this chapter
    # Added it here for convenience, so that this section
    # can be run as standalone without executing prior code
    # in the directory
    stop = stopwords.words('english')
    # next(stream_docs(path='movie_data.csv'))
    vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21, preprocessor=None, tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', random_state=1)
    doc_stream = stream_docs(path='movie_data.csv')
    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test2 = vect.transform(X_test)
    print("Accuracy: %.3f" % clf.score(X_test2, y_test))

    clf = clf.partial_fit(X_test2, y_test)
    a = 1
