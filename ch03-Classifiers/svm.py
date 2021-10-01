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


def maximum_margin_classification_with_SVM():
    """dealing with the nonlinearly separable case using slack variables"""
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idex=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def alternative_implementations_in_sklearn():
    ppn = SGDClassifier(loss='perceptron')
    lr = SGDClassifier(loss='log')
    svm = SGDClassifier(loss='hinge')


def using_kernel_trick_find_separating_hyperplanes(X_xor, y_xor):
    svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def solving_non_linear_problems_using_kernel_svm():
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1])
    y_xor = np.where(y_xor, 1, -1)
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    using_kernel_trick_find_separating_hyperplanes(X_xor, y_xor)


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

    maximum_margin_classification_with_SVM()

    alternative_implementations_in_sklearn()

    solving_non_linear_problems_using_kernel_svm()
