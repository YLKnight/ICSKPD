# M always means matrix
import numpy as np
from itertools import permutations


def error(C_hat, C, th=0):
    ''' Calculate Type I error and Power for region detection task '''
    ''' ---------- Input ---------- '''
    ''' C: matrix of true parameters '''
    ''' C_hat: matrix of estimated parameters with the same dimension of C '''
    ''' ---------- Output ---------- '''
    ''' typeI: Type I Error of C_hat '''
    ''' Power: Power of C_hat '''

    C_hat_ind = np.where(np.abs(C_hat) > th, 1, 0)
    C_ind = np.where(np.abs(C) != 0, 1, 0)
    typeI = np.sum(C_hat_ind * (1 - C_ind)) / np.sum(1 - C_ind)
    Power = np.sum(C_hat_ind * C_ind) / np.sum(C_ind)
    return typeI, Power


def Acc_cls(y, y_hat):
    ''' Calculate Accuracy for two-clustering task '''
    ''' ---------- Input ---------- '''
    ''' y - vector of true label'''
    ''' y_hat - vector of predicted label with the same length of y '''
    ''' ---------- Output ---------- '''
    ''' A scalar in [0, 1] - Accuracy of y_hat '''

    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    n = y.shape[0]
    freq = np.sum(y == y_hat) / n
    return max(freq, 1 - freq)


def Acc_pmtt(y, y_hat):
    ''' Calculate Accuracy for multi-clustering task '''
    ''' ---------- Input ---------- '''
    ''' y - vector of true numerical label'''
    ''' y_hat - vector of predicted numerical label with the same length of y '''
    ''' ---------- Output ---------- '''
    ''' A scalar in [0, 1] - Accuracy of y_hat '''

    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    n = y.shape[0]
    labels = np.unique(y)
    k = labels.shape[0]
    idx_lst = []
    for l in labels:
        idx_lst.append(np.where(y_hat == l)[0])
    perms = list(permutations(range(k)))
    y_perm_lst = []
    Acc_lst = []
    for p in perms:
        y_perm = np.zeros(n)
        for l in labels:
            y_perm[idx_lst[l]] = p[l]
        y_perm_lst.append(y_perm)
        Acc_lst.append(np.sum(y_perm == y))
    return max(Acc_lst) / n
