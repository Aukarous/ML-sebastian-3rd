"""
Supervised data compression via linear discriminant analysis
1) Principal component analysis versus linear discriminant analysis
2) The inner workings of linear discriminant analysis
3) Computing the scatter matrices
4) Calculate the mean vectors for each class:
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_within_class_scatter_matrix():
    d = 13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d))  # scatter matrix for each class
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
            class_scatter += (row - mv).dot((row - mv).T)

        S_W += class_scatter  # sum class scatter matrices

    print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))


def cov_matrix():
    print('Class label distribution: %s')
    d = 13  # number of features
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train == label].T)
        S_W += class_scatter

    print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))


def compute_between_class_scatter_matrix():



if __name__ == "__main__":
    df_wine = pd.read_csv('wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    np.set_printoptions(precision=4)
    mean_vecs = []

    # Standardizing the data.
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
        print('MV %s:%s\n' % (label, mean_vecs[label - 1]))

    # compute the within-class scatter matrix:
    compute_within_class_scatter_matrix()

    # cov_matrix
    cov_matrix()

    # compute the between-class scatter matrix:
    compute_between_class_scatter_matrix()