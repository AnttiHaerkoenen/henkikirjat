from pathlib import Path
import string

import cv2
import xmltodict

from rectangle import Rectangle


def get_ground_truth(
        image: Path,
        grid: Path,
):
    img = cv2.imread(str(image))
    data = xmltodict.parse(grid.read_text())
    regions = data['PcGts']['Page']['TableRegion']['TextRegion']
    for i, region in enumerate(regions):
        rect = Rectangle.from_dict(region)
        box = img[rect.np_slice]
        win_name = rect.id
        cv2.imshow(win_name, box)
        key = cv2.waitKeyEx(0)
        if 48 <= key <= 57:
            digit = str(key - 48)
            rect.content = digit
            regions[i] = rect.to_dict()
        cv2.destroyWindow(win_name)
    grid.write_text(xmltodict.unparse(
        data,
        pretty=True,
        short_empty_elements=True,
    ))


if __name__ == '__main__':
    im = Path('../data/test.jpg')
    grid = Path('../data/grids/test.xml')
    get_ground_truth(im, grid)












