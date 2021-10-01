"""
Using kernel principal component analysis for nonlinear mappings
"""

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy import __version__ as scipy_version
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from distutils.version import LooseVersion as Version


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.
    Args:
        X: {NumPy ndarray}, shape = [n_examples, n_features]
        gamma: float
            Tuning parameter of the RBF kernel
        n_components: int
          Number of principal components to return


    Returns:
        X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
            Projected dataset
    """
    # calculate pairwise squared Euclidean distances in the M * N dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # convert pairwise distance into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # collect the top K eigenvectors (projected examples)
    X_pc = np.column_stack([eigvecs[:, i] for i in range(n_components)])

    return X_pc


def example_1():
    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.show()

    scikit_pca = PCA(n_components=5)
    X_spca = scikit_pca.fit_transform(X)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    plt.show()

    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    plt.show()


def example_2():
    X, y = make_circles(n_samples=1000, rnadom_state=123, noise=0.1, factor=0.2)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.show()

    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='blue', marker='o', alpha=0.5)
    ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    plt.show()


def rbf_kernel_pca_2(X, gamma, n_components):
    """
    RBF kernel PCA implementation

    Args:
        X: {NumPy ndarray}, shape = [n_examples, n_features]
        gamma: float
            Tuning parameter of the RBF kernel
        n_components: int
            Number of principal components to return

    Returns:
        alphas: {NumPy ndarray}, shape = [n_examples, k_features]
           Projected dataset

        lambdas: list
           Eigenvalues

    """
    # calculate pairwise squared Euclidean distances in the M * N dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # collect the top K eigenvectors (projected examples)
    alphas = np.column_stack([eigvecs[:, i] for i in range(n_components)])

    # collect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


def project_new_data_points():
    X, y = make_moons(n_samples=100, random_state=123)
    alphas, lambdas = rbf_kernel_pca_2(X, gamma=15, n_components=1)
    x_new = X[25]
    x_proj = alphas[25]
    print(x_new)
    print(x_proj)

    # projection of the 'new' datapoint
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    plt.scatter(alphas[y == 0, 0], np.zeros(50), color='red', marker='^', alpha=0.5)
    plt.scatter(alphas[y == 1, 0], np.zeros(50), color='blue', marker='o', alpha=0.5)
    plt.scatter(x_proj, 0, color='black', label='Original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='green', label='Remapped point X[25]', marker='x', s=500)
    plt.yticks([], [])
    plt.legend(scatterpoints=1)

    plt.tight_layout()
    plt.show()


def kernel_pca_in_sklearn():
    X, y = make_moons(n_samples=100, random_state=123)
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)
    plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Implementing a kernel principal component analysis in python
    example_1()

    # separating concentric circles
    example_2()

    # projecting new data points
    project_new_data_points()

    # kernel principal component analysis in scikit-learn
    kernel_pca_in_sklearn()
