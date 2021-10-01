"""
Unsupervised dimensionality reduction via principal component analysis

The main steps behind principal component analysis

Extracting the principal components step-by-step
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version
from numpy import exp as np_exp
# from scipy import exp as sp_exp

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA


def lr_using_2_prin_com():
    """
    Training logistic regression classifier using the first 2 principal components
    """
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr = lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.xlabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    color=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)


def pca_in_sklearn():
    """
    Principle component analysis in scikit-learn
    """
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)
    # pca.explained_variance_ratio_
    plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.show()
    plt.close()

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()


if __name__ == "__main__":
    df_wine = pd.read_csv('wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    # Standardizing the data.
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    """
    Accidentally, I wrote `X_test_std = sc.fit_transform(X_test)` instead of 
                          `X_test_std = sc.transform(X_test)`. 
    In this case, it wouldn't make a big difference since the mean and standard deviation of the test set 
    should be (quite) similar to the training set. 
    However, as remember from Chapter 3, the correct way is to re-use parameters from the training set if
     we are doing any kind of transformation -- the test set should basically stand for "new, unseen" data.
    
    My initial typo reflects a common mistake is that some people are *not* re-using these parameters from 
    the model training/building and standardize the new data "from scratch." 
    
    Here's simple example to explain why this is a problem.
    Let's assume we have a simple training set consisting of 3 examples with 1 feature (let's call this 
        feature "length"):
    
        - train_1: 10 cm -> class_2
        - train_2: 20 cm -> class_2
        - train_3: 30 cm -> class_1
        mean: 20, std.: 8.2
    After standardization, the transformed feature values are
 
        - train_std_1: -1.21 -> class_2
        - train_std_2: 0 -> class_2
        - train_std_3: 1.21 -> class_1
    
    Next, let's assume our model has learned to classify examples with a standardized length value < 0.6 as 
    class_2 (class_1 otherwise). So far so good. 
    Now, let's say we have 3 unlabeled data points that we want to classify:
        - new_4: 5 cm -> class ?
        - new_5: 6 cm -> class ?
        - new_6: 7 cm -> class ?
 
    If we look at the "unstandardized "length" values in our training dataset, it is intuitive to say that 
    all of these examples are likely belonging to class_2. However, if we standardize these by re-computing 
    standard deviation and mean you would get similar values as before in the training set 
    and your classifier would (probably incorrectly) classify examples 4 and 5 as class 2.
        - new_std_4: -1.21 -> class 2
        - new_std_5: 0 -> class 2
        - new_std_6: 1.21 -> class 1
    However, if we use the parameters from your "training set standardization," we'd get the values:
        - example5: -18.37 -> class 2
        - example6: -17.15 -> class 2
        - example7: -15.92 -> class 2

    The values 5 cm, 6 cm, and 7 cm are much lower than anything we have seen in the training set previously
    Thus, it only makes sense that the standardized features of the "new examples" are much lower than 
    every standardized feature in the training set.
    """

    # Eigendecomposition of the covariance matrix
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    """
    Above, I used the numpy.linalg.eig function to decompose the symmetric covariance matrix into 
    its eigenvalues and eigenvectors.
    
    This is not really a "mistake," but probably suboptimal. It would be better to use numpy.linalg.eigh in 
    such cases, which has been designed for Hermetian matrices. The latter always returns real eigenvalues; 
    whereas the numerically less stable np.linalg.eig can decompose nonsymmetric square matrices, 
    you may find that it returns complex eigenvalues in certain cases. (S.R.)
    """

    """
    Total and explained variance
    """
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.close()

    """
    Feature transformation
    """
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals)))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    X_train_std[0].dot(w)
    X_train_pca = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0],
                    X_train_pca[y_train == l, 1],
                    c=c, label=l, marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    plt.close()

    pca_in_sklearn()

    lr_using_2_prin_com()
