from pathlib import Path
import json

import cv2

from src.template_matching.rectangle import Rectangle


def get_ground_truth(
        image: Path,
        grid: Path,
):
    img = cv2.imread(str(image))
    # todo convert to json grid
    data = xmltodict.parse(grid.read_text())
    regions = data['PcGts']['Page']['TableRegion']['TextRegion']
    for region in enumerate(regions):
        rect = Rectangle.from_xml_dict(region)
        box = img[rect.np_slice]
        win_name = rect.id
        cv2.imshow(win_name, box)
        key = cv2.waitKeyEx(0)
        if 48 <= key <= 57:
            digit = str(key - 48)
            rect.content = digit
            regions[i] = rect.to_xml_dict()
        cv2.destroyWindow(win_name)
    # todo write to grid


if __name__ == '__main__':
    im = Path('../data/test1.jpg')
    grid = Path('../data/grids/test.xml')
    get_ground_truth(im, grid)
