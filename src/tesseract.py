from pathlib import Path
import locale
import os

locale.setlocale(locale.LC_ALL, 'C')
from tesserocr import PyTessBaseAPI, RIL
from PIL import Image
import xmltodict

from rectangle import Rectangle
import img_tools


def predict_page_content(
        img_file,
        xml_file,
        data_dir,
):
    os.chdir(data_dir)
    img_path = Path(img_file)
    xml_path = Path(xml_file)

    with xml_path.open(encoding='utf-8') as fin, PyTessBaseAPI() as api:
        doc = xmltodict.parse(fin.read(), encoding='utf-8')
        api.SetImageFile(img_path.name)
        rects = [Rectangle.from_xml_dict(r) for r in doc['PcGts']['Page']['TableRegion']['TextRegion']]
        for r in rects[200:]:
            img_tools.view_rectangle(r, img_path.name)
        for r in rects:
            api.SetRectangle(r.x_min, r.y_min, r.w, r.h)
            r.predicted = api.GetUTF8Text().strip()
        doc['PcGts']['Page']['TableRegion']['TextRegion'] = [r.to_xml_dict() for r in rects]
        output = xmltodict.unparse(doc, pretty=True, short_empty_elements=True)
        xml_path.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    predict_page_content(r'Henkikir_3355R.jpg', r'./grids/data.pdf-2.xml', r'../data')
    # os.chdir(r'../data')
    # with PyTessBaseAPI() as api:
    #     img = Path(r'Henkikir_3355R.jpg')
    #     api.SetImageFile(img.name)
    #     cells = api.GetComponentImages(RIL.TEXTLINE, True)
    #     for i, (im, box, _, _) in enumerate(cells):
    #         api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
    #         print(f'{i}:' + api.GetUTF8Text())
