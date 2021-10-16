import os
import sys
import tarfile
import time
import urllib.request
import pyprind
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer

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
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn.decomposition import LatentDirichletAllocation


def reporthook(count, block_size, total_size):
    global start_time
    if count==0:
        start_time=time.time()
        return
    duration = time.time() - start_time
    progress_size=int(count*block_size)
    speed = progress_size/(1024.**2*duration)
    percent = count*block_size*100./total_size
    sys.stdout.write()
    sys.stdout.flush()


if __name__ == "__main__":
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
