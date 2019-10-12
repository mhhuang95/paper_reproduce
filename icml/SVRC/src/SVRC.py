#writen by Minhui Huang
#10/10/2019

#-----------------------------------------------------------------
"""
    Implementation of SVRC algorithms for the following paper
    "Stochastic Variance-Reduced Cubic Regularized Newton Methods"
    Dongruo Zhou, Pan Xu, Quanquan Gu
"""
#-----------------------------------------------------------------


import numpy as np


def cubic_solver(g, H, M):
    """
    Solver for the cubic subproblem

    :param g: Estimated gradient
    :param H: Estimated Hessian
    :param M: cubic penalty parameter
    :return:
    """
    






def svrc(grad, hess, n, d, bg, bh, M, S, T, x0=None, func=None, verbose=True):
    """
    SVRC algo for solvinng finite-sum problems
    :param grad: gradient function in the form of grad(x, idx), where idx is a list of induces
    :param hess: hessian function in the form of hess(x, idx), where idx is a list of induces
    :param n, d: size of the problem
    :param bg, bh: batch size for the gradient and hession
    :param M: cubic penalty parameter
    :param S: outer loop numbers
    :param T: inner loop numbers
    :param x0: starting point
    :param func: function value
    :return:
    """

    if x0 is None:
        x_hat = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x_hat = x0.copy()
    else:
        raise ValueError("x0 must be a numpy array of size (d, )")

    x_ts = x_hat.copy()
    for s in range(S):

        gs = grad(x_hat, range(n))
        hs = hess(x_hat, range(n))

        for t in range(T):
            Ig = np.random.choice(n, bg, replace=False)
            Ih = np.random.choice(n, bh, replace=False)
            v_ts = grad(x_ts, Ig) - grad(x_hat, Ig) + gs - (hess(x_hat, Ig) - hs).dot(x_ts - x_hat)
            U_ts = hess(x_ts, Ih) - hess(x_hat, Ih) + hs

            #solve cubic subproblem
            h_ts = cubic_solver(v_ts, U_ts, M)

            x_ts += h_ts

        x_hat = x_ts.copy()

    return x_hat


