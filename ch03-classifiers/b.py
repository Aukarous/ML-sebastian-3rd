"""
Modeling class probabilities via logistic regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cost_1(z):
    return -np.log(sigmoid(z))


def cost_0(z):
    return -np.log(1 - sigmoid(z))


def logistic_regression():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.show()


def learning_weights_of_logistic_cost_function():
    z = np.arange(-10,10,0.1)
    phi_z = sigmoid(z)
    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z,c1,label='J(w) if y=1')
    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z,c0,linestyle='--',label='J(w) if y=0')

    plt.ylim(0.0,5.1)
    plt.xlim([0,1])
    plt.xlabel('$\phi$(z)')
    plt.ylabel('J(w)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """ run separated """
    # Logistic regression intuition and conditional probabilities
    logistic_regression()

    # Learning the weights of the logistic cost function
    learning_weights_of_logistic_cost_function()

    #