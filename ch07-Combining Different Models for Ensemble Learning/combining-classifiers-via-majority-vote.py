# coding: utf-8
import math
import operator
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.pipeline

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.pipeline import _name_estimators

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='class_label', weights=None):
        """
        A majority vote ensemble classifier
        Args:
            classifiers: array-like, shape = [n_classifiers]
                Different classifiers for the ensemble
            vote: str, {'class_label', 'probability'} (default='class_label')
                If 'class_label' the prediction is based on the argmax of class labels. Else if 'probability',
                the argmax of the sum of probabilities is used to predict the class label
                (recommended for calibrated classifiers).
            weights: array-like, shape = [n_classifiers], optional (default=None)
                If a list of `int` or `float` values are provided, the classifiers are weighted by
                importance; Uses uniform weights if `weights=None`.
        """
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        self.label_enc_ = LabelEncoder()
        self.classifiers_ = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit classifiers.
        Args:
            X: {array-like, sparse matrix}, shape = [n_examples, n_features]
                Matrix of training examples.
            y: array-like, shape = [n_examples]
                Vector of target class labels.

        Returns:
            self: object
        """
        if self.vote not in ('probability', 'class_label'):
            raise ValueError("vote must be 'probability' or 'class_label'; got (vote=%r)" % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal got %d weights, %d classifiers' %
                             (len(self.weights), len(self.classifiers)))
        # Use LabelEncoder to ensure class labels start with 0, which is important for np.argmax call in self.predict
        self.label_enc_.fit(y)
        self.classes_ = self.label_enc_.classes_
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.label_enc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Args:
            X: {array-like, sparse matrix}, shape = [n_examples, n_features]
                Training vectors, where n_examples is the number of examples and n_features is the number of features.

        Returns:
            avg_proba : array-like, shape = [n_examples, n_classes]
                Weighted average probability for each class per example.
        """
        proba_s = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(proba_s, axis=0, weights=self.weights)
        return avg_proba

    def predict(self, X):
        """Predict class labels for X.

        Args:
            X: {array-like, sparse matrix}, shape = [n_examples, n_features]
                Matrix of training examples.

        Returns:
            maj_vote : array-like, shape = [n_examples]
                Predicted class labels.
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X))
        else:  # 'class_label' vote
            # collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1,
                                           arr=predictions)
            maj_vote = self.label_enc_.inverse_transform(maj_vote)

            return maj_vote

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=True)

        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s_%s' % (name, key)] = value
                return out


def simple_majority_vote_classifier():
    np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))
    ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
    p = np.average(ex, axis=0, weights=[0.2, 0.4, 0.6])
    print(np.argmax(p))


def evaluate_tuning_ensemble_classifier():
    colors = ['black', 'orange', 'blue','green']
    line_styles = [':', '--','-.','-']
    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, line_styles):
        # assuming the label of the positive class is 1
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]


if __name__ == "__main__":
    # Using the majority voting principle to make predictions
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    l_e = LabelEncoder()
    y = l_e.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

    clf1 = LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', random_state=1)

    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)

    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

    clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

    print('10-fold cross validation:\n')
    for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
        print('ROC AUC: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))

    # Majority Rule (hard) Voting
    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    clf_labels += ['Majority voting']
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
        print('ROC AUC: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))

    # Evaluating and tuning the ensemble classifier
    evaluate_tuning_ensemble_classifier()