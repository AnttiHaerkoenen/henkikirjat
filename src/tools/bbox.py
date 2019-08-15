import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.io import imread
from skimage.util import invert
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray


if __name__ == '__main__':
    image = imread('../../data/test1.jpg')
    image = invert(rgb2gray(image)[321:420, 10:100])

    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(5))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(bw)
    image_label_overlay = label2rgb(label_image, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        if 200 <= region.area <= 500:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
