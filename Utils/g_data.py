from g_shape import *
import numpy as np
from skimage import io as ios


def Gnrt_shape(img_shape_m=128, img_shape_n=128, shape="OneCircle", radius=None):
    canvas = np.zeros((img_shape_m, img_shape_n))
    # Circle shape
    if shape == "OneCircle":
        gnrt_shape = circle(canvas, 48, 80, radius)
        # gnrt_shape = circle(canvas, 40, 72, 5)

    if shape == "con_circle":
        gnrt_shape = con_circle(canvas, 48, 80, 8, 15)
    if shape == "rectangles":
        c2 = rectangle(canvas, 16, 16, 13)  # 10
        c2 = rectangle(c2, 48, 80, 7)
        c2 = rectangle(c2, 116, 44, 4)
        gnrt_shape = c2

    if shape == "ThreeCircles":
        c2 = circle(canvas, 48, 80, radius[0])
        c2 = circle(c2, 16, 16, radius[1])  # 10
        c2 = circle(c2, 100, 44, radius[2])
        # c2 = circle(c2, 112, 48, 4)
        gnrt_shape = c2

    if shape == "inten_circles":
        c2 = circle(canvas, 16, 16, 7)
        c2 = circle(c2, 116, 44, 4) * 2
        c2 = circle(c2, 48, 80, 13)
        gnrt_shape = c2

    if shape == "con_circles":
        c2 = circle(canvas, 16, 16, 7)
        c2 = con_circle(c2, 48, 80, 8, 15)
        c2 = circle(c2, 116, 44, 4) * 3
        gnrt_shape = c2

    # buterfly
    if shape == "Butterfly":
        bf = ios.imread("butterfly.png", as_gray=True)/255
        gnrt_shape = bf

    return gnrt_shape


def Gnrt_3D(dims, shape="ball", center=None, radius=None):
    canvas = np.zeros(dims)
    if shape == "ball":
        gnrt_shape = ball(canvas, center, radius)

    return gnrt_shape
