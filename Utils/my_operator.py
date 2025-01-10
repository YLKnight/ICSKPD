# M always means matrix
import numpy as np
from skimage import io as ios


def Vec(M):
    ''' Vectorize a matrix M '''
    return M.reshape(-1, 1)


def Vec_inv(M, m, n):
    ''' Transform a vector to a matrix '''
    return M.reshape(m, n)


def R_opt(M, idctshape):
    m, n = M.shape
    p1, p2 = idctshape
    assert m % p1 == 0 and n % p2 == 0, "Dimension wrong"
    d1, d2 = m // p1, n // p2
    RM = []
    for i in range(p1):
        for j in range(p2):
            Mij = M[d1*i: d1*(i + 1), d2*j: d2*(j + 1)]
            RM.append(Vec(Mij))
    return np.concatenate(RM, axis=1).T


def R_opt_pro(A, idctshape):
    m, n = A.shape
    p1, p2 = idctshape
    assert m % p1 == 0 and n % p2 == 0, "Dimension wrong"
    d1, d2 = m // p1, n // p2
    strides = A.itemsize * np.array([p2*d2*d1, d2, p2*d2, 1])
    A_blocked = np.lib.stride_tricks.as_strided(A, shape=(p1, p2, d1, d2), strides=strides)
    RA = A_blocked.reshape(-1, d1*d2)
    return RA


def R_inv(RC, m1, m2, n1, n2):
    ''' Inverse R operator for matrix: (m1*n1, m2*n2) to (m1*m2, n1*n2) '''
    m1n1, m2n2 = RC.shape
    C = np.zeros([m1 * m2, n1 * n2])
    bb = []
    for i in range(m1n1):
        Block = Vec_inv(RC[i, :], m2, n2)
        ith = i // n1  # quotient
        jth = i % n1  # remainder
        C[m2 * ith:m2 * (ith + 1), n2 * jth:n2 * (jth + 1)] = Block
        bb.append(Block)

    return C, bb


def soft_threshold(x, th):
    return np.sign(x) * np.maximum(np.abs(x) - th, 0)


def hard_threshold(x, th):
    return x * (abs(x) > th)


def num2dum(ary):
    ''' Numerical variable to dummy variable '''
    # For Example: [0, 1 ,2] to [[1, 0], [0, 1], [0, 0]]
    cats = np.sort(np.unique(ary))
    dum = np.zeros((ary.shape[0], cats.shape[0]-1))
    for i, c in enumerate(cats[:-1]):
        dum[np.where(ary == c)[0], i] = 1
    return dum


def dum2num(dum):
    ''' Dummy variable to numerical variable '''
    # For Example: [[1, 0], [0, 1], [0, 0]] to [0, 1 ,2]
    n, c = dum.shape
    ary = np.zeros(shape=(n, ), dtype=int) + c
    for i, c in enumerate(range(c)):
        ary[np.where(dum[:, c] == 1)] = i
    return ary


def Gnrt(n, p, pw, sigma_img, int_sgn, sigma_ns, seed=0):
    rng = np.random.RandomState(seed)
    # Set the size of images
    D1, D2 = np.power(2, pw), np.power(2, pw)
    # Generate theta 2 matrix
    theta2 = rng.normal(0, sigma_img, D1 * D2).reshape(D1, D2)
    # Generate difference matrix
    signal = ios.imread('../Figures/butterfly.png', as_gray=True) / 255
    diff = int_sgn * signal
    # Generate theta 1 matrix
    theta1 = theta2 + diff

    n1, n2 = int(n * p), int(n * (1 - p))
    Y = np.concatenate((np.repeat(1, n1), np.repeat(0, n2)))
    X_TS = np.zeros(shape=(n, D1, D2))
    for i in range(n):
        w = rng.normal(0, sigma_ns, D1 * D2).reshape(D1, D2)
        X_TS[i] = Y[i] * theta1 + (1 - Y[i]) * theta2 + w
    # X_bar = X_TS - np.mean(X_TS, axis=0)

    return theta1, theta2, diff, Y, X_TS


def Gnrt_3(n, pw, sigma_img, int_sgn, sigma_ns, seed=0):
    rng = np.random.RandomState(seed)
    # Set the size of images
    D1, D2 = np.power(2, pw), np.power(2, pw)
    # Generate theta 3 matrix
    theta3 = rng.normal(0, sigma_img, D1 * D2).reshape(D1, D2)
    # Generate difference matrix
    signal1 = ios.imread('../Figures/bicircle1.png', as_gray=True) / 255
    signal2 = ios.imread('../Figures/bicircle2.png', as_gray=True) / 255

    diff1 = int_sgn * signal1
    diff2 = int_sgn * signal2
    theta1 = theta3 + diff1
    theta2 = theta3 + diff2

    n1 = n2 = int(n / 3)
    n3 = n - n1 - n2
    Y_num = np.concatenate((np.repeat(0, n1), np.repeat(1, n2), np.repeat(2, n3)))
    Y = num2dum(Y_num)
    X_TS = np.zeros(shape=(n, D1, D2))
    for i in range(n):
        W = rng.normal(0, sigma_ns, D1 * D2).reshape(D1, D2)
        X_TS[i] = Y[i, 0] * theta1 + Y[i, 1] * theta2 + W

    return theta1, theta2, theta3, diff1, diff2, Y, X_TS