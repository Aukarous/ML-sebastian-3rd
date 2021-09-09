"""
Adaptive linear neurons and the convergence of learning

Minimizing cost functions with gradient descent

Implementing an adaptive linear neuron in python

docstrings: Google
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron_algo import plot_decision_regions


class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.cost = None
        self.cost_ = None

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """compute linear activation"""
        return X

    def fit(self, X, y):
        """
        Args:
            X: array-like, shape = [n_examples, n_features], training vectors, where n_examples is the
            number of examples and n_features is the number of features.
            y: array-like, shape = [n_examples], target values

        Returns:
        --------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            """
            Note that the "activation" method has no effect in the code since it is simply an identity 
                function.
            Could write 'output=self.net_input(X)' directly instead.
            The purpose of the activation is more conceptual, in the case of logistic regression (as we will
            see later), we could change it to a sigmoid function to implement a logistic regression 
                classifier.
            
            """
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


def implement_adaptive_linear_neuron(df_t):
    y = df_t.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df_t.iloc[0:100, [0, 2]].values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()


def improve_gradient_descent(df_t):
    """Improving gradient descent through feature scaling"""
    # standardize features
    y = df_t.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df_t.iloc[0:100, [0, 2]].values

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 0].std()
    ada_gd = AdalineGD(n_iter=15, eta=0.01)
    ada_gd.fit(X_std, y)

    plot_decision_regions(X_std,y,classifier_t=ada_gd)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada_gd.cost_)+1), ada_gd.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('URL:', s)

    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    # function implemented, can run separated.
    implement_adaptive_linear_neuron(df_t=df)
    improve_gradient_descent(df_t=df)
