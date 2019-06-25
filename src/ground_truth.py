from pathlib import Path
import string

import cv2
import xmltodict

from rectangle import Rectangle


def get_ground_truth(
        image: Path,
        grid: Path,
):
    with Image.open(img_file) as img:
        data = xmltodict.parse(grid.read_text())
        for region in data['PcGts']['Page']['TableRegion']['TextRegion']:
            rect = Rectangle.from_dict(region)
            box = img.crop(rect.pil_box)
            box.show()
            key = cv2.waitKey(0)
            if str(key) in string.digits:
                region['Content']['Unicode'] = key
        grid.write_text(xmltodict.unparse(
            data,
            pretty=True,
            short_empty_elements=True,
        ))


if __name__ == '__main__':
    im = Path('../data/3355_straight.jpg')
    grid = Path('../data/grids/3355_straight.xml')
    get_ground_truth(im, grid)
