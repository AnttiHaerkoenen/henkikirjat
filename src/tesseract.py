from pathlib import Path
import locale
import os

locale.setlocale(locale.LC_ALL, 'C')
from tesserocr import PyTessBaseAPI, RIL
from PIL import Image


if __name__ == '__main__':
    os.chdir(r'../data')
    with PyTessBaseAPI() as api:
        img = Path(r'Henkikir_3355R.jpg')
        api.SetImageFile(img.name)
        cells = api.GetComponentImages(RIL.TEXTLINE, True)
        for i, (im, box, _, _) in enumerate(cells):
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
            print(f'{i}:' + api.GetUTF8Text())
