from pathlib import Path
import locale
import os

locale.setlocale(locale.LC_ALL, 'C')
from tesserocr import PyTessBaseAPI, RIL
from PIL import Image
import xmltodict

from rectangle import Rectangle


def predict_page_content(
        *,
        img_file,
        xml_file,
        data_dir,
):
    os.chdir(data_dir)
    img_path = Path(img_file)
    xml_path = Path(xml_file)

    with xml_path.open() as fin, PyTessBaseAPI() as api:
        doc = xmltodict.parse(fin.read())
        api.SetImageFile(img_path.name)
        rects = [Rectangle.from_dict(r) for r in doc['PcGts']['Page']['TableRegion']['TextRegion']]
        for r in rects:
            api.SetRectangle(r.x_min, r.y_min, r.w, r.h)
            r.predicted = api.GetUTF8Text()
        doc['PcGts']['Page']['TableRegion']['TextRegion'] = [r.to_dict() for r in rects]
        output = xmltodict.unparse(doc, pretty=True) \
            .replace('></Coords>', '/>') \
            .replace('></RegionRefIndexed>', '/>')
        xml_path.write_text(output)


if __name__ == '__main__':
    os.chdir(r'../data')
    with PyTessBaseAPI() as api:
        img = Path(r'Henkikir_3355R.jpg')
        api.SetImageFile(img.name)
        cells = api.GetComponentImages(RIL.TEXTLINE, True)
        for i, (im, box, _, _) in enumerate(cells):
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
            print(f'{i}:' + api.GetUTF8Text())
