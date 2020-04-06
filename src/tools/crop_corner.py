import sys

from PIL import Image


def crop_corner(
        img: str,
        corner: str = 'LR',
):
    image = Image.open(img)
    w, h = image.size

    coords = {
        'UL': (0, w // 2, 0, h // 2),
        'UR': (w // 2, w, 0, h // 2),
        'LR': (w // 2, w, h // 2, h),
        'LL': (0, w // 2, h // 2, h),
    }

    image = image.crop(coords.get(corner, (0, w, 0, h)))
    image.save(f'cropped_{img}')


if __name__ == '__main__':
    _, corner, *images = sys.argv

    for image in images:
        crop_corner(image, corner)
