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


def get_digit_locations(
        image_file,
        data_dir,
        template_dir,
):
    os.chdir(data_dir)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.imread(image_file, cv2.IMREAD_COLOR)

    edges = cv2.Canny(image, 400, 1000)
    cv2.imwrite('canny.jpg', edges)

    template = cv2.imread('5.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = template.shape

    res = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.29
    loc = np.where(res >= threshold)


if __name__ == '__main__':
    os.chdir('../data')
    image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

    edges = cv2.Canny(image, 400, 1000)
    cv2.imwrite('canny.jpg', edges)

    template = cv2.imread('5.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = template.shape

    res = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.29
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('res.jpg', image_rgb)

    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite('thresh.jpg', thresh)

    # cnts = cv2.findContours(
    #     thresh.copy(),
    #     cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE
    # )
    # cnts = imutils.grab_contours(cnts)
    # output = image.copy()
    # for i, c in enumerate(cnts):
    #     cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    #     cv2.imwrite(f"contour{i}.jpg", output)
    #     print(f"Contour file {i} written")
