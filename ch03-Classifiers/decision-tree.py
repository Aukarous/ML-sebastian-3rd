import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch
from pydotplus import graph_from_dot_data
from a import plot_decision_regions


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return p * np.log2(p) - (1 - p) * np.log2(1 - p)


def error(p):
    return 1 - np.max([p, 1 - p])


def maximizing_information_gain():
    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
                             ['Entropy', 'Entropy (scaled)', 'Gini impurity', 'Misclassification error'],
                             ['-', '-', '--', '-.'],
                             ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('impurity index')
    plt.show()


def building_decision_tree():
    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree_model.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=tree_model, test_idx=range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    tree.plot_tree(tree_model)
    plt.show()

    dot_data = export_graphviz(tree_model, filled=True, rounded=True, class_names=['Setosa',
                                                                                   'Versicolor',
                                                                                   'Virginica'],
                               feature_names=['petal length', 'petal width'], out_file=None)
    graph = graph_from_dot_data(dot_data)


def strong_learners_via_rf():
    forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    maximizing_information_gain()

    iris_data: Bunch = datasets.load_iris()
    X = iris_data.data[:, [2, 3]]
    y = iris_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    """building a decision tree"""
    building_decision_tree()

    """combining weak to strong learners via random forests"""
    strong_learners_via_rf()
