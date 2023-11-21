# These functions can be used to fit the models.
# You can change anything you want and you can use or add your own functions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def poly_basis_transform(X, k):
    """
    X: one dimensional input array
    k: order of polynomial

    returns
    PHI: data matrix with polynomial features
    """
    X = np.array(X)
    m = X.size
    PHI = np.ones((m, k + 1))
    for i in range(k + 1):
        PHI[:, i] = np.power(X, i)
    return PHI


def mse(PHI, y, w):
    """
    PHI: augmented feature matrix
    y:   output vector
    w:   weight vector

    returns mean squared error
    """
    m = PHI.shape[0]
    return 1 / (2 * m) * np.linalg.norm(PHI.dot(w) - y) ** 2


def linear_eval(PHI, w):
    """
    PHI: augmented feature matrix
    w:   weight vector

    returns: array of linear hypothesis evaluations h(x) for all examples x in PHI
    """
    return PHI.dot(w)


def linear_regression_fit(PHI, y, n_iter, eta, lamb=0, w_init=None):
    """
    PHI:    augmented feature matrix
    y:      output vector
    n_iter: number of gradient descent iterations
    eta:    learning rate
    lamb:   regularization parameter
    w_init: initial weights

    returns
    w:    fitted weight vector
    loss: list of MSE values for each iteration
    """
    (m, k) = PHI.shape

    if w_init is None:
        w_init = 0.1 * np.random.randn(k)

    w = w_init
    loss = [mse(PHI, y, w)]  # first entry of loss is MSE with respect to initial weights

    for i in range(n_iter):
        w0 = w.copy()
        w0[0] = 0  # exclude bias from regularization
        grad = 1 / m * (PHI.T.dot(PHI.dot(w) - y) + lamb * w0)
        w = w - eta * grad
        loss.append(mse(PHI, y, w))

    return w, loss
