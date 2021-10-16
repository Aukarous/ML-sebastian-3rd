import os
import sys
import tarfile
import time
import urllib.request
import pyprind
import re

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

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
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import LatentDirichletAllocation


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emotions).replace('-', ''))
    return text


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    """波特词干提取算法，可能是最古老且最简单的词干提取算法，其他常见的词干提取算法包括Snowball stemmer和Lancaster stemmer
    """
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def remove_stop_words(text):
    """不能用来区分不同类别文档的有用信息的单词，如is，and，has，like等，移除停用词可能对处理原始或者归一化的词频而非tf-idf有益，因为已经降低了那些频繁
    出现的单词的权重。
    """
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    res = [w for w in tokenizer_porter(text) if w not in stop]
    return res


def introducing_BOW_model(df_t):
    """Introducing the bag-of-words model
    By calling the fit_transform method on CountVectorizer, we just constructed the vocabulary of the bag-of-words
    model and transformed the following three sentences into sparse feature vectors:
        1. The sun is shining
        2. The weather is sweet
        3. The sun is shining, the weather is sweet, and one and one is two
    """
    # Transforming documents into feature vectors
    count = CountVectorizer()
    docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
    bag = count.fit_transform(docs)
    # print the contents of the vocabulary to get a better understanding of the underlying concepts
    print(count.vocabulary_)
    print(bag.toarray())
    np.set_printoptions(precision=2)

    # Scikit-learn implements the `TfidfTransformer`, that takes the raw term frequencies from
    # `CountVectorizer` as input and transforms them into tf-idfs:
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

    tf_is = 3
    n_docs = 3
    idf_is = np.log((n_docs + 1) / (3 + 1))
    tfidf_is = tf_is * (idf_is + 1)
    print('tf_idf of term "is" = %.2f' % (tfidf_is))

    df['review_processed'] = df['review'].apply(preprocessor)
    # 准备好电影评论数据集后，考虑如何将文本语料库拆分成独立的元素，tokenize文件的一种方法是通过把清洗后的文档沿空白字符拆分成单独的单词


def lr_doc_cls(df_t):
    # Strip HTML and punctuation to speed up the GridSearch later
    from nltk.corpus import stopwords
    stop = stopwords.words('english')

    X_train = df_t.loc[:25000, 'review'].values
    y_train = df_t.loc[:25000, 'sentiment'].values
    X_test = df_t.loc[25000:, 'review'].values
    y_test = df_t.loc[25000:, 'sentiment'].values
    # 👇 使用TfidfVectorizer 替换 CountVectorizer 和 TfidfTransformer，其中的TfidfTransformer包含了转换器对象
    tf_idf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},  # 👈 逆正则化参数C，比较正则化强度
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf': [False],
                   'vect__norm': [None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  ]
    lr_tfidf = Pipeline([('vect', tf_idf), ('clf', LogisticRegression(random_state=0, solver='liblinear'))])
    # 👇 由于特征向量和词汇很多，网格搜索的计算成本相当昂贵，因此限制了参数组合的数量
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)
    clf = gs_lr_tfidf.best_estimator_
    print('Best parameter set:%s' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


def illustrate_k_fold():
    np.random.seed(0)
    np.set_printoptions(precision=6)
    y = [np.random.randint(3) for i in range(25)]
    X = (y + np.random.randn(25)).reshape(-1, 1)
    cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False, random_state=0).split(X, y))
    lr = LogisticRegression(random_state=123, multi_class='ovr', solver='lbfgs')
    score = cross_val_score(lr, X, y, cv=cv5_idx)
    print("cross_val_score:{}".format(score))
    """ By executing the code above, we created a simple data set of random integers that shall represent our class 
    labels. 
    Next, we fed the indices of 5 cross-validation folds (`cv3_idx`) to the `cross_val_score` scorer, 
    which returned 5 accuracy scores -- these are the 5 accuracy values for the 5 test folds.  
    Next, let us use the `GridSearchCV` object and feed it the same 5 cross-validation sets 
    (via the pre-generated `cv3_idx` indices): """
    lr = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=1)
    gs = GridSearchCV(lr, {}, cv=cv5_idx, verbose=3).fit(X, y)
    """
    As we can see, the scores for the 5 folds are exactly the same as the ones from `cross_val_score` earlier.
    Now, the best_score_ attribute of the `GridSearchCV` object, which becomes available after `fit`ting, returns the 
    average accuracy score of the best model: """
    print("gs.best_score is :{}".format(gs.best_score_))
    """👆 As we can see, the result above is consistent with the average score computed the `cross_val_score`."""
    lr = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=1)
    score2 = cross_val_score(lr, X, y, cv=cv5_idx).mean()


if __name__ == "__main__":
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    # introducing_BOW_model(df)
    lr_doc_cls(df)
    """
    注意：lr_doc_cls函数中，`gs_lr_tfidf.best_score_`是k-折叠cross-validation得分均值。如果object：`GridSearchCV`是5-fold交叉验证（像上
    面的例子），则"best_score_"返回的平均分数超过最佳模型，可以通过下面的例子来解释
    """
    illustrate_k_fold()
