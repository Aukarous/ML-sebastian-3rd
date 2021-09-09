from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from a import plot_decision_regions


def lr_with_sklearn():
    """
    Training a logistic regression model with scikit-learn
    """
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
    lr.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    lr.predict_proba(X_test_std[:3, :])
    lr.predict_proba(X_test_std[:3, :]).sum(axis=1)
    lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
    lr.predict(X_test_std[:3, :])
    lr.predict(X_test_std[0, :].reshape(1, -1))


def tackle_overfit_via_regularization():
    weights, params = [], []
    for c in np.arange(-5, 5):
        lr = LogisticRegression(C=10. ** c, random_state=1, solver='lbfgs', multi_class='ovr')
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10. ** c)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    # run separated
    lr_with_sklearn()

    tackle_overfit_via_regularization()
