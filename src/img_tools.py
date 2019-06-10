from PIL import Image

from rectangle import Rectangle


def view_rectangle(rect: Rectangle, img_file: str):
    with Image.open(img_file) as img:
        box = img.crop(rect.pil_box)
        box.show()


if __name__ == '__main__':
    pass
