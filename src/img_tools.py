import os

from PIL import Image
import cv2
import imutils
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
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('thresh.jpg', thresh)

    # cnts = cv2.findContours(
    #     thresh.copy(),
    #     cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE
    # )
    # cnts = imutils.grab_contours(cnts)
    # output = image.copy()
    # # loop over the contours
    # for i, c in enumerate(cnts):
    #     # draw each contour on the output image with a 3px thick purple
    #     # outline, then display the output contours one at a time
    #     cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    #     cv2.imwrite(f"contour{i}.jpg", output)
    #     print(f"Contour file {i} written")
