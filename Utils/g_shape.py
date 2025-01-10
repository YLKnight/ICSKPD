import numpy as np
import numpy.linalg as la
import copy


def rectangle(img, x0, y0, r):
    temp = img.copy()
    m, n = temp.shape
    temp[int(x0 - r):int(x0 + r), int(y0 - r):int(y0 + r)] = 1
    return temp


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


## generate concentric circle
def con_circle(img, x0, y0, r1, r2):
    temp = copy.deepcopy(img)
    m, n = temp.shape
    for i in range(m):
        for j in range(n):
            dist = np.round(la.norm(np.array([i - x0, j - y0]), 2))
            if dist >= r1 and dist <= r2:
                temp[i, j] = 1

    return temp


def _T(img, x0, y0, length, width):
    m, n = img.shape
    r1 = int(length / 2)
    r2 = int(width / 2)
    img[int(x0):int(x0 + length * 0.8), int(y0 - r2):int(y0 + r2)] = 1
    img[int(x0 - width):int(x0), int(y0 - r1):int(y0 + r1)] = 1
    return img


def _cross(img, x0, y0, length, width):
    m, n = img.shape
    r1 = int(length / 2)
    r2 = int(width / 2)
    img[int(x0 - r1 - r2):int(x0 + r1 - r2), int(y0 + r1 - r2):int(y0 + r1 + r2)] = 1
    img[int(x0 - width):int(x0), int(y0):int(y0 + length)] = 1
    return img


def triangle(img, x0, y0, r):
    m, n = img.shape
    x0 = x0
    y0 = y0
    h = np.round(np.sqrt(3) / 2 * r)
    x1, y1 = x0 - r / 2, y0
    x2, y2 = x0 + r / 2, y0
    e1 = (np.array([x0 - x1, y0 - y1]) / la.norm(np.array([x0 - x1, y0 - y1]), 2)).reshape(-1, 1)
    e2 = (np.array([x0 - x2, y0 - y2]) / la.norm(np.array([x0 - x2, y0 - y2]), 2)).reshape(-1, 1)

    for i in range(m):
        for j in range(n):
            if i <= y0:
                vec1 = np.array([i - x1, j - y1]).reshape(-1, 1)
                dist1 = np.round(la.norm(vec1, 2))
                _norm1 = vec1 / dist1
                vec2 = np.array([i - x2, j - y2]).reshape(-1, 1)
                dist2 = la.norm(vec2, 2)
                _norm2 = np.round(vec2 / dist2)
                _theta1 = np.trace(_norm1.T.dot(e1))
                _theta2 = np.trace(_norm2.T.dot(e2))
                if _theta1 >= np.sqrt(2) / 2 and _theta2 >= np.sqrt(2) / 2 and (np.abs(i - x1) != np.abs(j - y0)):
                    img[i, j] = 1
    return img


def shift_circle(img, x0, y0, r, n, intensity=5):
    # center is (x0,y0)
    # intensity
    np.random.seed(539)
    img_list = []
    shift_x = np.sort(np.random.normal(size=n)) * intensity
    shift_y = np.sort(np.random.normal(size=n)) * intensity

    for i in range(n):
        tmp = circle(img, x0 + shift_x[i], y0 + shift_y[i], r)
        img_list.append(tmp)
    return img_list


def shift_circles(img, c_ls, r_ls, n, intensity_ls):
    num = len(c_ls)
    img_list = []
    shift_ls = np.random.uniform(low=-1, high=1, size=2*num*n).reshape(n, 2*num)
    for i in range(n):
        tmp = img
        shift = shift_ls[i, :]
        for c in range(num):
            x0, y0 = c_ls[c]
            r, intensity = r_ls[c], intensity_ls[c]
            shift_x, shift_y = shift[(2*c):2*(c+1)] * intensity
            tmp = circle(tmp, x0 + shift_x, y0 + shift_y, r)
        img_list.append(tmp)

    return img_list


def shift_butterfly(img, n, intensity=10):
    row, col = np.where(img > 0)
    # complete shift pairs
    shift_x = np.arange(-intensity, intensity+1)
    shift_y = np.arange(-intensity, intensity+1)
    shift = []
    for i in range(len(shift_x)):
        for j in range(len(shift_y)):
            shift.append(np.array([shift_x[i], shift_y[j]]))
    # sample shift pairs
    if n < len(shift):
        ind = np.sort(np.random.choice(range(len(shift)), n, replace=False))
    else:
        ind = np.sort(np.random.choice(range(len(shift)), n))
    shift = np.array(shift)
    shift_smp = shift[ind, :]

    img_list = []
    for i in range(n):
        tmp = np.zeros_like(img)
        tmp[row + shift_smp[i, 0], col + shift_smp[i, 1]] = 1
        img_list.append(tmp)
    return img_list
