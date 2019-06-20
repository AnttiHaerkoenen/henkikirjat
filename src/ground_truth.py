from pathlib import Path

from img_tools import view_rectangle


def get_ground_truth(
        image: Path,
        grid: Path,
):
    with image.open() as im, grid.open('rw') as grid:
        pass


if __name__ == '__main__':
    pass
