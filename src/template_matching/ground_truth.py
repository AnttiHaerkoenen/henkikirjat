from pathlib import Path
import json

import cv2

from src.template_matching.rectangle import Rectangle


def get_ground_truth(
        image: Path,
        image_id: str,
        grid_path: Path,
):
    img = cv2.imread(str(image))
    data = json.loads(grid_path.read_text())
    regions = data[image_id]

    i = 0
    digits = []
    while True:
        if not 0 <= i < len(regions):
            break
        rect = Rectangle.from_json_dict(regions[i])
        box = img[rect.np_slice]
        win_name = rect.rect_id
        cv2.imshow(win_name, box)
        key = cv2.waitKeyEx(0)
        if key == 97:  # a
            i -= 1
            digits = []
        elif key == 100:  # d:
            i += 1
            digits = []
        elif key == 13:  # enter
            if len(digits) > 0:
                rect.content = ''.join(digits)
                regions[i] = rect.to_json_dict()
            i += 1
            digits = []
        elif 48 <= key <= 57:
            digits.append(str(key - 48))
        elif key in (27, 98):  # exit, b
            break
        cv2.destroyAllWindows()
    data[image_id] = regions
    grid_path.write_text(json.dumps(data, indent=4))


if __name__ == '__main__':
    im = Path('../../data/test1.jpg')
    grid = Path('../../data/grids/test.json')
    get_ground_truth(im, 'test1', grid)
