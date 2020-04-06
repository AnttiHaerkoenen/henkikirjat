import sys
import os

import cv2
import numpy as np


def pad_image(img, pad, h_new, w_new):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    if h > h_new or w > w_new:
        raise ValueError("New size must be larger than old")
    top, r = divmod(h_new - h, 2)
    bottom = top + r
    left, r = divmod(w_new - w, 2)
    right = left + r
    pad_width = (top, bottom), (left, right)
    padded = np.pad(image, pad_width=pad_width, mode='constant', constant_values=pad)
    cv2.imwrite(img, padded)


if __name__ == '__main__':
    _, pad, h, w, *images = sys.argv
    for img in images:
        pad_image(img, float(pad), int(h), int(w))
