import math
from functools import reduce

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from skimage.morphology import binary_dilation
from skimage.color import rgb2gray
from skimage.util import invert
from skimage.io import imread
from skimage.draw import line
from skimage.filters import threshold_otsu
from skimage.transform import hough_line


def remove_lines(
        img: np.ndarray,
        threshold: int = 30,
) -> np.ndarray:
    theta_v = np.linspace(-0.05, 0.05, 10)
    theta_h = theta_v + np.pi / 2
    theta_both = np.concatenate((theta_v, theta_h))
    h, theta, d = hough_line(img, theta=theta_both)
    width, height = img.shape

    lines = []

    for i, dist in enumerate(d):
        for j, angle in enumerate(theta):
            if threshold < h[i, j]:
                # todo fix
                # points = [
                #     (0, int(dist // np.sin(angle))),
                #     (width, int((dist - width * np.cos(angle)) // np.sin(angle))),
                #     (int(dist // np.cos(angle)), 0),
                #     (int((dist - height * np.sin(angle)) // np.cos(angle)), height),
                # ]
                points = [
                    (int(dist // np.sin(angle)), 0),
                    (int((dist - width * np.cos(angle)) // np.sin(angle)), width),
                    (0, int(dist // np.cos(angle))),
                    (height, int((dist - height * np.sin(angle)) // np.cos(angle))),
                ]
                points = [p for p in points if (0 <= p[0] < width and 0 <= p[1] < height)]
                if len(points) == 2:
                    p1, p2 = points
                    rr, cc = line(*p1, *p2)
                    line_img = np.zeros_like(img)
                    line_img[rr, cc] = 1
                    lines.append(line_img)
    mask = reduce(np.maximum, lines)
    mask = binary_dilation(mask)
    # todo masking
    return img


def plot_lines(
        img: np.ndarray,
        threshold: int = 30,
):
    theta_v = np.linspace(-0.05, 0.05, 10)
    theta_h = theta_v + np.pi / 2
    theta_both = np.concatenate((theta_v, theta_h))
    h, theta, d = hough_line(img, theta=theta_both)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Original')
    ax[0].set_axis_off()

    ax[1].imshow(img, cmap=cm.gray)

    for i, dist in enumerate(d):
        for j, angle in enumerate(theta):
            if threshold < h[i, j]:
                if angle < np.pi / 4:
                    x0, x1 = 0, img.shape[1]
                    y0 = dist / np.sin(angle)
                    y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
                    ax[1].plot((x0, x1), (y0, y1), '-r')
                else:
                    y0, y1 = 0, img.shape[0]
                    x0 = dist / np.cos(angle)
                    x1 = (dist - y1 * np.sin(angle)) / np.cos(angle)
                    ax[1].plot((x0, x1), (y0, y1), '-r')
    ax[1].set_xlim((0, img.shape[1]))
    ax[1].set_ylim((img.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected grid lines')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image = imread('../../data/test1.jpg')
    image = rgb2gray(image)
    # box = image[321:420, 10:100]
    box = image[488:555, 1197:1301]
    thresh = threshold_otsu(box)
    box = box > thresh
    box = invert(box)
    plot_lines(box)
    box = remove_lines(box)
    plot_lines(box)
