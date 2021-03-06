import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """
    Perceptron classifier.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        # for _ in range(self.n_iter):
        #     errors=0
        #     for xi,target in zip(X,y):
        #         update = self.eta*(target-self.predi)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# v1 = np.array([1, 2, 3])
# v2 = 0.5 * v1
# np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def select_serosa_and_versicolor(df):
    """ select setosa and versicolor """
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


def training_perception_model(df):
    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Numbers of updates')
    plt.show()


def plot_decision_regions(X_f, y_f, classifier_t, resolution=0.02):
    """ plotting decision regions """
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_f))])

    # plot the decision surface
    x1_min, x1_max = X_f[:, 0].min() - 1, X_f[:, 0].max() + 1
    x2_min, x2_max = X_f[:, 1].min() - 1, X_f[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier_t.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y_f)):
        plt.scatter(x=X_f[y_f == cl, 0],
                    y=X_f[y_f == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    # read iris data
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(s, header=None, encoding='utf-8')
    # ?????????????????????????????????????????????????????????below function can run separatedly??????????????????????????????????????????????????????
    select_serosa_and_versicolor(df=df)

    # training_perception_model(df=df)

    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    classifier = Perceptron(eta=0.1, n_iter=10)
    classifier.fit(X, y)
    plot_decision_regions(X, y, classifier, resolution=0.02)
    # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
