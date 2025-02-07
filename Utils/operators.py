# M always means matrix
import numpy as np


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


def Vec(M):
    ''' Vectorize a matrix M '''
    return M.reshape(-1, 1)


def Vec_inv(M, shape):
    ''' Transform a vector to a matrix '''
    return M.reshape(shape)


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


def Rearrange_T(T, Adim, Bdim):
    ''' R operator for tensor: (p1*d1, p2*d2, p3*d3) to (p1*p2*p3, d1*d2*d3) '''
    N1, N2, N3 = T.shape
    p1, p2, p3 = Adim
    d1, d2, d3 = Bdim
    RC = []
    assert N1 == p1 * d1 and N2 == p2 * d2 and N3 == p3 * d3, 'Dimension wrong!'
    for i in range(p1):
        for j in range(p2):
            for k in range(p3):
                Tij = T[d1 * i:d1 * (i + 1), d2 * j:d2 * (j + 1), d3 * k:d3 * (k + 1)]
                RC.append(Vec(Tij))
    return np.concatenate(RC, axis=1).T


def R_inv_T(RT, Adim, Bdim):
    ''' Inverse R operator for tensor: (p1*p2*p3, d1*d2*d3) to (p1*d1, p2*d2, p3*d3) '''
    P, D = RT.shape
    p1, p2, p3 = Adim
    d1, d2, d3 = Bdim
    assert P == p1 * p2 * p3 and D == d1 * d2 * d3, 'Dimension wrong!'
    slices = []
    fibers = []
    for i in range(P):
        fiber = RT[i, ].reshape(Bdim)
        fibers.append(fiber)
        if len(fibers) == p2:
            slice = np.concatenate(fibers, axis=1)
            slices.append(slice)
            fibers = []
    T = np.concatenate(slices, axis=0)
    return T


def Rearrange_4D(T, Adim, Bdim):
    ''' R operator for tensor: (p1*d1, p2*d2, p3*d3, p4*d4) to (p1*p2*p3*p4, d1*d2*d3*d4) '''
    N1, N2, N3, N4 = T.shape
    p1, p2, p3, p4 = Adim
    d1, d2, d3, d4 = Bdim
    RC = []
    assert N1 == p1 * d1 and N2 == p2 * d2 and N3 == p3 * d3 and N4 == p4 * d4, 'Dimension wrong!'
    for i in range(p1):
        for j in range(p2):
            for k in range(p3):
                for l in range(p4):
                    Tij = T[d1 * i:d1 * (i + 1), d2 * j:d2 * (j + 1), d3 * k:d3 * (k + 1), d4 * l:d4 * (l + 1)]
                    RC.append(Vec(Tij))
    return np.concatenate(RC, axis=1).T


def R_inv_4D(RT, Adim, Bdim):
    ''' Inverse R operator for tensor: (p1*p2*p3*p4, d1*d2*d3*d4) to (p1*d1, p2*d2, p3*d3, p4*d4) '''
    P, D = RT.shape
    p1, p2, p3, p4 = Adim
    d1, d2, d3, d4 = Bdim
    assert P == p1 * p2 * p3 * p4 and D == d1 * d2 * d3 * d4, 'Dimension wrong!'
    fourds = []
    tensors = []
    slices = []
    fibers = []
    for i in range(P):
        fiber = RT[i, ].reshape(Bdim)
        fibers.append(fiber)
        if len(fibers) == p4:
            slice = np.concatenate(fibers, axis=3)
            slices.append(slice)
            fibers = []
        if len(slices) == p3:
            tensor = np.concatenate(slices, axis=2)
            tensors.append(tensor)
            slices = []
        if len(tensors) == p2:
            fourd = np.concatenate(tensors, axis=1)
            fourds.append(fourd)
            tensors = []
        print(i, len(fibers), len(slices), len(tensors), len(fourds))
    T = np.concatenate(fourds, axis=0)
    return T