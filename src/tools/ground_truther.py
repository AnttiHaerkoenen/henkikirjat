import os
import glob
from pathlib import Path
import json

import cv2
from skimage.io import imread
from skimage.util import invert

from src.tools.digits import DigitFilter, extract_digits, clip_numbers


def save_ground_truth(
        *,
        digits,
        image=None,
        img_file,
        json_file,
):
    if image is None:
        image = imread(img_file)

    outf = Path(json_file)

    i = 0
    characters = [None for _ in digits]

    while True:
        if not 0 <= i < len(digits):
            break

        bbox = digits[i].bbox
        win_name = f"Digit {i} at {bbox[0]}, {bbox[1]} ({img_file.split('/')[-1]})"
        minr, minc, maxr, maxc = bbox
        box = image[minr:maxr, minc:maxc]

        cv2.imshow(win_name, box)
        key = cv2.waitKeyEx(0)
        if key == 97:  # a
            i -= 1
        elif key == 100:  # d
            i += 1
        elif key == 101:  # e
            characters[i] = None
            i += 1
        elif 48 <= key <= 57:  # digits
            characters[i] = str(key - 48)
            i += 1
        elif key in (27, 98):  # exit, b
            break
        cv2.destroyAllWindows()

    data = [[digit.bbox, characters[i]] for i, digit in enumerate(digits)]
    outf.write_text(json.dumps(data, indent=True))


if __name__ == '__main__':
    year = 1905
    os.chdir('../../data/train')
    img_files = sorted(glob.iglob(f'./{year}/*.jpg'))
    digit_filter = DigitFilter(
        min_area=50,
        max_area=500,
        min_width=20,
        min_height=20,
        max_width=100,
        max_height=100,
    )
    jsons = [js.split('/')[-1] for js in glob.iglob(f'./{year}/labels/*.json')]
    for img in img_files:
        img_name = img.split('/')[-1]
        json_file = f"{img_name.split('.')[0]}_truth.json"
        if json_file in jsons:
            continue

        print(img_name)
        image = clip_numbers(
            img,
            f'../headers/plot_header_{year}.jpg',
            f'../headers/taxpayer_header_{year}.jpg',
            col_height=2350,
            plot_col_width=82,
            pop_col_width=1375,
        )
        h, w = image.shape
        image = invert(image)
        digits = extract_digits(image, 600, None, digit_filter, do_closing=True)
        save_ground_truth(
            digits=digits,
            image=image,
            img_file=img,
            json_file=f"./{year}/labels/{json_file}",
        )
        print(f"{img_name} saved.")
