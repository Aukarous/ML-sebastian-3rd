"""
Large scale machine learning and stochastic gradient descent
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron_algo import plot_decision_regions


class AdalineSGD(object):
    """Adaptive linear neuron classifier.
    Args:
        eta: float
            Learning rate(between 0.0 and 1.0）
        n_iter: int
            Passes over the training dataset.
        shuffle: bool(default: True)
            Shuffles training data every epoch if True to prevent cycles.
        random_state: int
            Random number generator seed for random weight initialization.


    Attributes
        w_: 1d-array
            Weights after fitting.
        cost_: list
            Sum-of-squares cost function value averaged over all training examples in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        self.cost_ = None

    def _initialize_weights(self, m):
        """initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _shuffle(self, X, y):
        """shuffle training data"""
        res = self.rgen.permutation(len(y))
        return X[res], y[res]

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, X):
        return X

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def fit(self, X, y):
        """ Fit training data.
        Args:
            X: {array-like}, shape=[n_examples, n_features]
                Training vectors, where n_examples is the number of examples and n_features is the number of
                features.
            y: array-like, shape=[n_examples]
                Target values.


        Returns:
            self: object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


if __name__ == "__main__":
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(s, header=None, encoding='utf-8')
    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # standardize features
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier_t=ada_sgd)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')

    plt.tight_layout()
    plt.show()
