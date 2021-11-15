import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def fitting_a_robust_regression_model_using_RANSAC(data, X, y):
    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=50,
                             loss='absolute_loss',
                             residual_threshold=5.0,
                             random_state=0
                             )
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='black', lw=2)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.legend(loc='upper left')
    plt.show()
    print('Sloop: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)


def evaluate_the_performance_of_linear_regression_models(data):
    X = data.iloc[:, :-1].values
    y = data['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)
    ary = np.array(range(100000))
    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white',
                label='Test data')

    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.show()

    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                           mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                                           r2_score(y_test, y_test_pred)))


def regularize_methods_for_regression(data):
    X = data.iloc[:, :-1].values
    y = data['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    lasso = Lasso(alpha=0.1)
    # ridge = Ridge(alpha=1.0)
    # elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print("lasso.coef_ : {}".format(lasso.coef_))
    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                           mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                                           r2_score(y_test, y_test_pred)))


def nonlinear_relationships(data):
    X = data[['LSTAT']].values
    y = data['MEDV'].values

    regr = LinearRegression()

    # create quadratic features
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # fit features
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))

    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))

    # plot results
    plt.scatter(X, y, label='Training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit,
             label='Linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2,
             linestyle=':')

    plt.plot(X_fit, y_quad_fit,
             label='Quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
             color='red',
             lw=2,
             linestyle='-')

    plt.plot(X_fit, y_cubic_fit,
             label='Cubic (d=3), $R^2=%.2f$' % cubic_r2,
             color='green',
             lw=2,
             linestyle='--')
    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('% Price in $1000s [MEDV]')
    plt.legend(loc='upper right')
    plt.show()


def transforming_dataset(data):
    X = data[['LSTAT']].values
    y = data['MEDV'].values
    regr = LinearRegression()

    # transform features
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)

    # fit features
    X_fit = np.arange(X_log.min() - 1, X_log.max() + 1)[:, np.newaxis]

    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

    # plot results
    plt.scatter(X_log, y_sqrt, label='Training points', color='lightgray')
    plt.plot(X_fit, y_lin_fit, label='Linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue', lw=2)
    plt.xlabel('log(% lower status of the population [LSTAT])')
    plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()


def lin_regplot(X_t, y_t, model):
    plt.scatter(X_t, y_t, c='steelblue', edgecolors='white', s=70)
    plt.plot(X_t, model.predict(X_t), color='black', lw=2)
    return


def dealing_with_nonlinear_relationships_using_random_forests(data):
    # decision tree regression
    X = data[['LSTAT']].values
    y = data['MEDV'].values
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)

    sort_idx = X.flatten().argsort()
    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.show()

    # random forest regression
    X = data.iloc[:, :-1].values
    y = data['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    forest = RandomForestRegressor(n_estimators=1000,
                                   criterion='mse',
                                   random_state=1,
                                   n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                           mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                                           r2_score(y_test, y_test_pred)))
    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', edgecolors='white', marker='o', s=35, alpha=0.9,
                label='Training dataZ')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', edgecolors='white', marker='s', s=35, alpha=0.9,
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("housing.data.txt", header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                  'MEDV']

    X = df[['RM']].values
    y = df['MEDV'].values
    # fitting_a_robust_regression_model_using_RANSAC(df, X, y)
    # evaluate_the_performance_of_linear_regression_models(df)
    # regularize_methods_for_regression(df)
    # nonlinear_relationships(df)
    # transforming_dataset(df)
    dealing_with_nonlinear_relationships_using_random_forests(df)
