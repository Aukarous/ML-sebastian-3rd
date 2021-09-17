"""
Using kernel principal component analysis for nonlinear mappings
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version
from numpy import exp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from scipy import exp
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA


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
    X,y = make_circles(n_samples=1000, rnadom_state=123, noise=0.1, factor=0.2)
    plt.scatter(X[y==0,0],X[y==0,1],color='red',marker='^',alpha=0.5)
    plt.scatter(X[y==1,0],X[y==1,1],color='blue',marker='o',alpha=0.5)
    plt.tight_layout()
    plt.show()

    X_kpca = rbf_kernel_pca(X,gamma=15,n_components=2)
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))
    ax[0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],color='blue',marker='o',alpha=0.5)
    ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y==0,0],np.zeros((500,1))+0.02,color='red',marker='^',alpha=0.5)
    ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    plt.show()


def project_new_data_points():



if __name__ == "__main__":
    # Implementing a kernel principal component analysis in python
    example_1()

    # separating concentric circles
    example_2()

    # projecting new data points
    project_new_data_points()