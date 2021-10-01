"""
Selecting linear discriminants for the new feature subspace
Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$:
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from dimen_reduc_via_pca import plot_decision_regions


def sort_eigenvectors(eigen_vals, eigen_vecs):
    # make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    """
    visually confirm that the list is correctly sorted by decreasing eigenvalues 
    """
    print('Eigenvalues in descending order:\n')
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    tot = sum(eigen_vals.real)
    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)
    plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='Individual "discriminability')
    plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative "discriminability"')
    plt.ylabel('"Discriminability" ratio')
    plt.xlabel('Linear discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                   eigen_pairs[1][1][:, np.newaxis].real))
    print('Matrix W:\n', w)

    """ 
    projecting examples onto the new feature space
    """
    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train == l, 0],
                    X_train_lda[y_train == l, 1] * (-1),
                    c=c, label=l, marker=m)

    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    """
    LDA via scikit-learn
    """
    lda = LDA(n_components=2)
    X_train_lad = lda.fit_transform(X_train_std, y_train)
    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr = lr.fit(X_train_lda, y_train)
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    X_test_lda = lda.transform(X_test_std)
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def solve_generalized_eigenvalue():
    d = 13
    S_B = np.zeros((d, d))
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train == label].T)
        S_W + class_scatter
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    """
    Above, I used the [`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.
    linalg.eig.html) function to decompose the symmetric covariance matrix into its eigenvalues and 
    eigenvectors.
    This is not really a "mistake," but probably suboptimal. It would be better to use 
    [`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html) 
    in such cases, which has been designed for 
    [Hermetian matrices](https://en.wikipedia.org/wiki/Hermitian_matrix). 
    The latter always returns real  eigenvalues; whereas the numerically less stable `np.linalg.eig` can 
    decompose nonsymmetric square matrices, 
    you may find that it returns complex eigenvalues in certain cases. (S.R.)

    """

    # Sort eigenvectors in descending order of the eigenvalues:
    sort_eigenvectors(eigen_vals, eigen_vecs)


if __name__ == "__main__":
    df_wine = pd.read_csv('wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    np.set_printoptions(precision=4)
    mean_vecs = []

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_test)
    X_test_std = sc.transform(X_test)

    # Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$
    solve_generalized_eigenvalue()
