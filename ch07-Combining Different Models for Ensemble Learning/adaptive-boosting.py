# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    csv_path = os.path.join(os.getcwd(), 'wine.data')
    df_wine = pd.read_csv(csv_path, header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                       'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    # drop 1 class
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # How boosting works
    # Applying adaboost using scikit-learn

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)
    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)

    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)

    ada_train = accuracy_score(y_train, y_train_pred)
    ada_test = accuracy_score(y_test, y_test_pred)
    print('Adaboost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, ax_arr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8, 3))
    for idx, clf_t, tt in zip([0, 1], [tree, ada], ['Decision tree', 'AdaBoost']):
        clf_t.fit(X_train, y_train)
        Z = clf_t.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax_arr[idx].contourf(xx, yy, Z, alpha=0.3)
        ax_arr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='^')
        ax_arr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', marker='o')
        ax_arr[idx].set_title(tt)

    ax_arr[0].set_ylabel('Alcohol', fontsize=12)
    plt.tight_layout()
    plt.text(0, -0.2, s='OD280/OD315 of diluted wines', ha='center', va='center', fontsize=12,
             transform=ax_arr[1].transAxes)
    plt.show()
