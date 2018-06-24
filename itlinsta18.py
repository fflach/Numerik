""""
Created: 17.05.2018

@author: Edda Eich-Soellner

Zweck: 
"""
import numpy as np
import scipy as sp

from scipy import sparse as sps
import scipy.linalg as spla



def asparse(n):
    I = []
    J = []
    V = []
    mat = [(0, 0, 12), (0, 1, -6), (0, 2, 4 / 3)]
    for i in range(1, n-1):
        if i - 2 >= 0:
            mat.append((i, i - 2, 1))
        mat.append((i, i - 1, -4))
        mat.append((i, i + 1, -4))
        mat.append((i, i, 6))
        if i + 2 <= n-1:
            mat.append((i, i + 2, 1))
    mat.append((n-1, n - 3, 4 / 3))
    mat.append((n-1, n - 2, -6))
    mat.append((n-1, n-1, 12))
    for elem in mat:
        I.append(elem[0])
        J.append(elem[1])
        V.append(elem[2])

    A = sps.coo_matrix((V, (I, J)), shape=(n , n ))
    return A


def jacobi_beam(y, c):
    """
    Jacobi adapted for Euler-Bernoulli-beam
    One iteration only
    y: approximation to solution
    c: right hand side
    return: next iterate
    """
    # BEGIN SOLUTION
    n = len(y)
    z = np.zeros(n)
    z[0] = (c[0] - (-6 * y[1] + 4 / 3 * y[2])) / 12
    z[1] = (c[1] - (-4 * y[0] - 4 * y[2] + y[3])) / 6
    for i in range(2, n - 2):
        z[i] = (c[i] - (y[i - 2] - 4 * y[i - 1] - 4 * y[i + 1] + y[i + 2])) / 6
    z[n - 2] = (c[n - 2] - (y[n - 4] - 4 * y[n - 3] - 4 * y[n - 1])) / 6
    z[n - 1] = (c[n - 1] - (4 / 3 * y[n - 3] - 6 * y[n - 2])) / 12
    return z
    # END SOLUTION


def gs_beam(y, c):
    """
    Gauss-Seidel adapted for Euler-Bernoulli-beam
    One iteration only
    y: approximation to solution
    c: right hand side
    return: next iterate
    """
    # BEGIN SOLUTION
    n = len(y)
    y[0] = (c[0] - (-6 * y[1] + 4 / 3 * y[2])) / 12
    y[1] = (c[1] - (-4 * y[0] - 4 * y[2] + y[3])) / 6
    for i in range(2, n - 2):
        y[i] = (c[i] - (y[i - 2] - 4 * y[i - 1] - 4 * y[i + 1] + y[i + 2])) / 6
    y[n - 2] = (c[n - 2] - (y[n - 4] - 4 * y[n - 3] - 4 * y[n - 1])) / 6
    y[n - 1] = (c[n - 1] - (4 / 3 * y[n - 3] - 6 * y[n - 2])) / 12
    return y
    # END SOLUTION


def multi_it(method, n, c, tol=1e-5, itmax=1000):
    """
    iteration loop over method (gauss seidel or jacobi)
    method: gs or jacobi
    n: system dimension
    c: right hand side
    tol: tolerance (increment)
    itmax: maximum number of iterations
    return:
        y: last iterate
        it: number of iterations used so far
        enorm: norm of increment
        kfac: Konvergenzrate
    """
    # BEGIN SOLUTION
    enorm = 2 * tol
    enormold = 2 * enorm
    y = np.zeros(n)
    it = 0
    while (enorm > tol and it < itmax):
        yold = y.copy()
        y = method(y, c)
        enormold = enorm
        enorm = np.linalg.norm(y - yold)
        kfac = enorm / enormold
        it += 1
    return y, it, enorm, kfac
    # END SOLUTION


import numpy as np

# define global constants b, d, L, E, I, g, rho load
d = 0.05
b = 0.1
L = 15
E = 2E11
I = b * d ** 3 / 12
g = 9.81
rho = 7850


def f(x):
    """
    load function, right hand side f
    """
    # BEGIN SOLUTION
    return -rho * b * d * g
    # END SOLUTION


def yex(x):
    return f(x) * x ** 2. * (L - x) ** 2 / (24 * E * I)


def cg(Amaly, c, y, itmax=10, eps=1e-5):
    """
    in:
    Amaly: Funktion, die A*y effizient für dieses konkrete Problem berechnet
    c: rechte Seite
    y: Startlösung
    itmax: maximal zulässige Anzahl Iterationen
    eps: Toleranz für die Norm des Residuums
    return:
    yn: Lösung bzw. aktuelle Iterierte bei Nichtkonvergenz
    it: Anzahl verwendeter Iterationen
    r: Residuum
    """
    # BEGIN SOLUTION
    n = len(y)
    iouter = 0
    imax = min(itmax, n)
    xn, it, r, erg = cgmaxn(amaly, c, y, itmax=imax, eps=eps)

    while (spla.norm(r) > eps and iouter < itmax / n):
        # print('Restart ', xn)
        xn, it, r, erg = cgmaxn(amaly, c, xn, itmax=imax, eps=eps)
        iouter += 1

    return xn, it, r


def amaly(y):
    n = len(y)
    Ax = np.zeros(n)
    Ax[0] = 12 * y[0] - 6 * y[1] + 4 / 3 * y[2]
    Ax[1] = -4 * y[0] + 6 * y[1] - 4 * y[2] + y[3]
    for i in range(2, n - 2):
        Ax[i] = y[i - 2] - 4 * y[i - 1] + 6 * y[i] - 4 * y[i + 1] + y[i + 2]
    Ax[n - 2] = y[n - 4] - 4 * y[n - 3] + 6 * y[n - 2] - 4 * y[n - 1]
    Ax[n - 1] = 4 / 3 * y[n - 3] - 6 * y[n - 2] + 12 * y[n - 1]
    return Ax


def cgmaxn(amaly, c, x, itmax=10, eps=1e-5):
    """

    """

    r = - amaly(x) + c
    d = r.copy()
    xn = x
    rs = np.dot(r, r)
    i = 0
    erg = []
    while (spla.norm(r) > eps and i <= itmax):
        i += 1
        z = amaly(d)
        alpha = rs / np.dot(d, z)
        xn = xn + alpha * d
        rn = r - alpha * z
        rns = np.dot(rn, rn)
        beta = rns / rs
        d = rn + beta * d
        r = rn
        rs = rns
        erg.append([xn, r])
        # print(i)
        # print(r)
    return xn, i, r, erg
