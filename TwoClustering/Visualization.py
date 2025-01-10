# matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
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


def Vis_mean_diff(theta1, theta2, diff, pw_block):
    plt.figure(figsize=(10, 3.5))
    plt.suptitle('Latent mean images for 2 clusters')
    plt.subplot(131)
    plt.title('Theta 1')
    gridplot(theta1, pw_block, grid=False)
    plt.xlabel('Mean with signal')
    plt.subplot(132)
    plt.title('Theta 2')
    gridplot(theta2, pw_block, grid=False)
    plt.xlabel('Mean without signal')
    plt.subplot(133)
    plt.title('Signal')
    gridplot(diff, pw_block, grid=True)
    plt.show()


def Vis_samples(mat1, mat2, mat3):
    sticks = False
    grid = False
    fz = 15
    cmap = 'gray'
    fig = plt.figure(figsize=(7, 3))
    plt.suptitle('Samples of 2 clusters for example', fontsize=fz + 2)
    plt.subplot(131)
    plt.title('A sample with signal', fontsize=fz)
    Vis(mat1, sticks=sticks, grid=grid, cmap=cmap)
    plt.axis('off')
    plt.subplot(132)
    plt.title('A sample without signal', fontsize=fz)
    Vis(mat2, sticks=sticks, grid=grid, cmap=cmap)
    plt.axis('off')
    plt.subplot(133)
    plt.title('True signal', fontsize=fz)
    Vis(mat3, sticks=sticks, grid=grid, cmap=cmap)
    plt.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()


def Vis_detection(A_hat, B_hat, C_hat, d2):
    plt.figure(figsize=(15, 4))
    # plt.suptitle('Region Detection')
    plt.subplot(131)
    plt.title(r'$\hat{A}$')
    sns.heatmap(A_hat, center=0)
    plt.subplot(132)
    plt.title(r'$\hat{B}$')
    sns.heatmap(B_hat, center=0)
    plt.yticks(np.arange(0, d2 + 1))
    plt.subplot(133)
    plt.title(r'$\hat{C} = \hat{A} \otimes \hat{B}$')
    sns.heatmap(C_hat, center=0)
    plt.show()
