import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread
from skimage.util import invert
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray

from src.tools.line_finder import remove_lines


if __name__ == '__main__':
    image = imread('../../data/test1.jpg')
    h, w, _ = image.shape
    image = invert(rgb2gray(image))[0:h // 2, 0:w // 2]

    thresh = threshold_yen(image)
    bw = image > thresh
    bw = remove_lines(bw, threshold=400)

    cleared = clear_border(bw)

    label_image = label(bw)

    for region in regionprops(label_image):
        if region.area <= 100:
            for (r, c) in region.coords:
                bw[r, c] = 0

    label_image = label(bw)
    image_label_overlay = label2rgb(label_image, image=bw)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
