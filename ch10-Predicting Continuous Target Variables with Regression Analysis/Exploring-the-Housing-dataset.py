import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# pd.set_option('')


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.cost_ = None

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            cost = (errors ** 2).sum() / 2.0
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return self.net_input(X)


def scatter_lr(data, x_std, y_std):
    lr = LinearRegressionGD()
    lr.fit(x_std, y_std)
    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
    plt.xlabel('Epoch')
    plt.ylabel('SSE')
    plt.tight_layout()
    plt.show()


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000s [MEDV] (standardized)')
    plt.show()

    print('Slope: %.3f' % (model.w_[1]))
    print('Intercept:%.3f' % (model.w_[0]))

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    num_rooms_std = sc_x.transform(np.array([[5.0]]))
    price_std = model.predict(num_rooms_std)
    print('Price in $1000s: %.3f' % sc_y.inverse_transform(price_std))


def visualize_important_characteristics(data):
    """ 相关矩阵与协方差矩阵再标准化特征计算方面保持一致
    相关矩阵是包含皮尔逊积矩相关系数（皮尔逊r）的方阵，用来度量特征对之间的线性依赖关系"""
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    scatterplotmatrix(df[cols].values, figsize=(10, 6), names=cols, alpha=0.5)
    plt.show()  # 散点图矩阵展示了数据集内部特征之间的关系

    # heatmap
    cm = np.corrcoef(data[cols].values.T)
    hm = heatmap(cm, row_names=cols, column_names=cols)
    plt.show()


def estimating_coefficient(X, y):
    slr = LinearRegression()
    slr.fit(X, y)
    y_pred = slr.predict(X)
    print('Sloop: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)

    plt.scatter(X, y, c='steelblue', edgecolors='white', s=70)
    plt.plot(X, slr.predict(X), color='black', lw=2)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.show()

    # Normal equations alternative
    # adding a column vector of 'ones'
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    print('Sloop: %.3f' % w[1])
    print('Intercept: %.3f' % w[0])


def implement_ordinary_least_squares_lr(data):
    # solving regression for regression parameters with gradient descent
    X = data[['RM']].values
    y = data['MEDV'].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()  # newaxis
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)
    scatter_lr(data, x_std=X_std, y_std=y_std)
    lin_regplot(X_std, y_std, lr)
    estimating_coefficient(X, y)


if __name__ == "__main__":
    df = pd.read_csv("housing.data.txt", header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                  'MEDV']
    visualize_important_characteristics(df)
    # scatter_lr(df)
    # lin_regplot(df)
    implement_ordinary_least_squares_lr(df)
