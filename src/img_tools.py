import os

from PIL import Image
import cv2
import numpy as np
from skimage import morphology

from rectangle import Rectangle


def view_rectangle(rect: Rectangle, img_file: str):
    with Image.open(img_file) as img:
        box = img.crop(rect.pil_box)
        box.show()


if __name__ == '__main__':
    os.chdir('../data')
    image = cv2.imread('test.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cv2.imwrite('test.jpg', contours)
