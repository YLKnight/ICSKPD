import numpy as np
from scipy.spatial import distance_matrix

from Utils.operators import *
from skimage import io as ios
import numpy.linalg as la
import copy


def circle(img, x0, y0, r):
    temp = copy.deepcopy(img)
    m, n = temp.shape
    for i in range(m):
        for j in range(n):
            dist = np.round(la.norm(np.array([i - x0, j - y0]), 2))
            if dist <= r:
                temp[i, j] = 1
    return temp


def ball(img, center, radius):
    temp = copy.deepcopy(img)
    N1, N2, N3 = temp.shape
    x, y, z = center
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                dist = np.round(la.norm(np.array([i - x, j - y, k - z]), 2))
                if dist <= radius:
                    temp[i, j, k] = 1
    return temp


def Gnrt_shape(img_shape_m=128, img_shape_n=128, shape="OneCircle", radius=None):
    canvas = np.zeros((img_shape_m, img_shape_n))
    if shape == "OneCircle":
        gnrt_shape = circle(canvas, 48, 80, radius)
        # gnrt_shape = circle(canvas, 40, 72, 5)

    if shape == "ThreeCircles":
        c2 = circle(canvas, 48, 80, radius[0])
        c2 = circle(c2, 16, 16, radius[1])  # 10
        c2 = circle(c2, 100, 44, radius[2])
        # c2 = circle(c2, 112, 48, 4)
        gnrt_shape = c2

    if shape == "Butterfly":
        bf = ios.imread("butterfly.png", as_gray=True) / 255
        gnrt_shape = bf

    return gnrt_shape


def peak(distances=np.ones(shape=(128, 128))):
    # M = 1 / (dist + 1)

    M = 1 / np.exp(distances)

    return M



def Gnrt_3D(dims, shape="ball", center=None, radius=None):
    canvas = np.zeros(dims)
    if shape == "ball":
        gnrt_shape = ball(canvas, center, radius)

    return gnrt_shape


def Generate_data_normal(signal, n=1000, positive_rate=0.5, sigma_of_image=1, sigma_of_noise=1, seed=0):
    rng = np.random.RandomState(seed)
    shape = signal.shape

    theta2 = rng.normal(0, sigma_of_image, shape)
    theta1 = theta2 + signal

    n1, n2 = int(n * positive_rate), int(n * (1 - positive_rate))
    Y = np.concatenate((np.repeat(1, n1), np.repeat(0, n2)))
    W = rng.normal(0, sigma_of_noise, (n, *shape))
    X_TS = np.array([Y[i] * theta1 + (1 - Y[i]) * theta2 + W[i] for i in range(n)])

    return theta1, theta2, Y, X_TS


def Generate_data_t(signal, n=1000, positive_rate=0.5, df=5, sigma_of_noise=1, seed=0):
    rng = np.random.RandomState(seed)
    shape = signal.shape

    theta2 = rng.standard_t(df=df, size=shape)
    theta1 = theta2 + signal

    n1, n2 = int(n * positive_rate), int(n * (1 - positive_rate))
    Y = np.concatenate((np.repeat(1, n1), np.repeat(0, n2)))
    W = rng.standard_t(df=df, size=(n, *shape)) * sigma_of_noise
    X_TS = np.array([Y[i] * theta1 + (1 - Y[i]) * theta2 + W[i] for i in range(n)])

    return theta1, theta2, Y, X_TS


def Generate_data_log_normal(signal, n=1000, positive_rate=0.5, sigma_of_image=1, sigma_of_noise=1, seed=0):
    rng = np.random.RandomState(seed)
    shape = signal.shape

    theta2 = rng.lognormal(0, sigma_of_image, size=shape)
    theta1 = theta2 + signal

    n1, n2 = int(n * positive_rate), int(n * (1 - positive_rate))
    Y = np.concatenate((np.repeat(1, n1), np.repeat(0, n2)))
    W = rng.lognormal(0, sigma_of_noise, size=(n, *shape))
    X_TS = np.array([Y[i] * theta1 + (1 - Y[i]) * theta2 + W[i] for i in range(n)])

    return theta1, theta2, Y, X_TS


def Generate_data_peak(n=1000, image_shape=(128, 128), positive_rate=0.5, sigma_of_noise=1, seed=0):
    rng = np.random.RandomState(seed)

    d1, d2 = image_shape
    center = np.array([0.5, 0.5])
    distances = np.zeros(shape=(d1, d2))
    for i in range(d1):
        for j in range(d2):
            position = np.array([i / d1, j / d2])
            distances[i, j] = np.linalg.norm(position - center)

    mask = circle(np.zeros(image_shape), int(d1 / 2), int(d2 / 2), 60)

    alphas_neg = rng.randint(1, 6, size=n - int(n * positive_rate))
    alphas_pos = rng.randint(6, 11, size=int(n * positive_rate))
    alphas = np.concatenate((alphas_neg, alphas_pos))
    Y = (alphas > 5).astype(int)
    X_TS = []
    Noise_TS = rng.normal(0, sigma_of_noise, size=(n, *image_shape))
    for alpha in alphas:
        X = peak(alpha * distances) * mask
        X_TS.append(X)

    signal = mask
    X_TS = np.array(X_TS) + Noise_TS
    return signal, Y, X_TS


def Generate_3_peak(n=1000, image_shape=(128, 128), sigma_of_noise=1, seed=0):
    rng = np.random.RandomState(seed)

    d1, d2 = image_shape
    center = np.array([0.5, 0.5])
    distances = np.zeros(shape=(d1, d2))
    for i in range(d1):
        for j in range(d2):
            position = np.array([i / d1, j / d2])
            distances[i, j] = np.linalg.norm(position - center)

    mask = circle(np.zeros(image_shape), int(d1 / 2), int(d2 / 2), 60)
    n1, n2 = int(n / 3), int(n / 3)

    alphas_1 = rng.randint(1, 4, size=n1)
    alphas_2 = rng.randint(4, 7, size=n2)
    alphas_3 = rng.randint(7, 10, size=n-n1-n2)
    alphas = np.concatenate((alphas_1, alphas_2, alphas_3))
    Y_num = (alphas - 1) // 3
    Y = num2dum(Y_num)
    X_TS = []
    Noise_TS = rng.normal(0, sigma_of_noise, size=(n, *image_shape))
    for alpha in alphas:
        X = peak(alpha * distances) * mask
        X_TS.append(X)

    signal = mask
    X_TS = np.array(X_TS) + Noise_TS
    return signal, Y, X_TS


def Generate_3(n, pw, sigma_img, sigma_ns, seed=0):
    rng = np.random.RandomState(seed)
    D1, D2 = np.power(2, pw), np.power(2, pw)
    theta3 = rng.normal(0, sigma_img, D1 * D2).reshape(D1, D2)
    canvas = np.zeros((D1, D2))
    x = y = 40
    R1, R2 = 8, 8
    signal1, signal2 = circle(canvas, x, y, R1), circle(canvas, D1 - x, D2 - y, R2)

    theta1 = theta3 + signal1
    theta2 = theta3 + signal2

    n1 = n2 = int(n / 3)
    n3 = n - n1 - n2
    Y_num = np.concatenate((np.repeat(0, n1), np.repeat(1, n2), np.repeat(2, n3)))
    Y = num2dum(Y_num)
    X_TS = np.zeros(shape=(n, D1, D2))
    for i in range(n):
        W = rng.normal(0, sigma_ns, D1 * D2).reshape(D1, D2)
        X_TS[i] = Y[i, 0] * theta1 + Y[i, 1] * theta2 + W

    return theta1, theta2, theta3, signal1, signal2, Y, X_TS


def Generate_3_log_normal(n, pw, sigma_img, sigma_ns, seed=0):
    rng = np.random.RandomState(seed)
    D1, D2 = np.power(2, pw), np.power(2, pw)
    theta3 = rng.lognormal(0, sigma_img, D1 * D2).reshape(D1, D2)
    canvas = np.zeros((D1, D2))
    x = y = 40
    R1, R2 = 8, 8
    signal1, signal2 = circle(canvas, x, y, R1), circle(canvas, D1 - x, D2 - y, R2)

    theta1 = theta3 + signal1
    theta2 = theta3 + signal2

    n1 = n2 = int(n / 3)
    n3 = n - n1 - n2
    Y_num = np.concatenate((np.repeat(0, n1), np.repeat(1, n2), np.repeat(2, n3)))
    Y = num2dum(Y_num)
    X_TS = np.zeros(shape=(n, D1, D2))
    for i in range(n):
        W = rng.lognormal(0, sigma_ns, D1 * D2).reshape(D1, D2)
        X_TS[i] = Y[i, 0] * theta1 + Y[i, 1] * theta2 + W

    return theta1, theta2, theta3, signal1, signal2, Y, X_TS


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(50, 10))
    # alphas = np.concatenate([np.arange(1, 11) / 10, np.array([1.5]), np.arange(2, 11)])
    # for i, alpha in enumerate(alphas):
    #     M = peak(alpha=alpha)
    #     plt.subplot(2, 10, i + 1)
    #     plt.imshow(M)
    #     plt.title(r'$\alpha = {:.1f}$'.format(alpha))
    # plt.show()

    # Y, X_TS = Generate_data_peak(1000, (128, 128))
