# matplotlib.use('agg')
import matplotlib.pyplot as plt
from Utils.my_operator import *


def gridplot(mat, p, grid=True):
    plt.imshow(mat)
    plt.xticks(np.arange(0, mat.shape[0]+1, np.power(2, p)))
    plt.yticks(np.arange(0, mat.shape[1]+1, np.power(2, p)))
    if grid:
        plt.grid(linestyle='-.', linewidth=0.5, which="both")


def Vis(mat, p=3, sticks=True, grid=True, cmap=None):
    if cmap is None:
        plt.imshow(mat)
    else:
        plt.imshow(mat, cmap=cmap)
    if sticks:
        plt.xticks(np.arange(0, mat.shape[0]+1, np.power(2, p)))
        plt.yticks(np.arange(0, mat.shape[1]+1, np.power(2, p)))
    else:
        plt.xticks([])
        plt.yticks([])
    if grid:
        plt.grid(linestyle='-.', linewidth=0.5, which="both")


def Vis_mean_diff(theta3, diff1, diff2, pw_block, D1, d1):
    horizon = np.arange(0, D1 + 1, d1)
    ticks = [' ' for _ in range(horizon.shape[0])]
    plt.figure(figsize=(10, 3.5))
    plt.suptitle('Signals for 2 clusters Compare to Control')
    plt.subplot(131)
    plt.title('Theta 3')
    gridplot(theta3, pw_block, grid=False)
    plt.xlabel('Control')
    plt.xticks(horizon, ticks)
    plt.subplot(132)
    plt.title('Difference 1')
    gridplot(diff1, pw_block, grid=True)
    plt.xticks(horizon, ticks)
    plt.subplot(133)
    plt.title('Difference 2')
    gridplot(diff2, pw_block, grid=True)
    plt.xticks(horizon, ticks)
    plt.show()


def Vis_samples(mat1, mat2, mat3, gridsize):
    plt.figure(figsize=(12, 4))
    plt.suptitle('Samples of 2 clusters for example')
    plt.subplot(131)
    plt.title('A Sample in Cluster 1')
    gridplot(mat1, gridsize, grid=False)
    plt.subplot(132)
    plt.title('A Sample in Cluster 2')
    gridplot(mat2, gridsize, grid=False)
    plt.subplot(133)
    plt.title('A Sample in Cluster 3')
    gridplot(mat3, gridsize, grid=False)
    plt.show()


def Vis_detection(Union, Union_hat, pw_block):
    sticks = False
    grid = False
    cmap = 'gray'
    fig = plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.title('True Signal')
    Vis(Union, pw_block, sticks=sticks, grid=grid, cmap=cmap)
    plt.subplot(122)
    plt.title('Region Detection')
    Vis(np.abs(Union_hat), pw_block, sticks=sticks, grid=grid, cmap=cmap)
    plt.show()
