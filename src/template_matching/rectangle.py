from collections import OrderedDict
from dataclasses import dataclass
from typing import Union, List


@dataclass(init=False)
class Rectangle:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    rect_id: Union[str, None] = None
    content: Union[str, None] = None
    predicted: Union[List[List], None] = None

    def __init__(
            self,
            x_min,
            y_min,
            x_max,
            y_max,
            rect_id=None,
            content=None,
            predicted=None,
    ):
        if x_max < x_min:
            raise ValueError("Impossible shape! x_max must be greater than x_min")
        if y_max < y_min:
            raise ValueError("Impossible shape! y_max must be greater than y_min")
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        self.rect_id = str(rect_id)
        self.content = content
        self.predicted = predicted

    @property
    def h(self):
        return self.y_max - self.y_min

    @property
    def w(self):
        return self.x_max - self.x_min

    @property
    def np_slice(self):
        return slice(self.y_min, self.y_max), slice(self.x_min, self.x_max)

    @property
    def coords(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def xml_coords(self) -> str:
        return f"{self.x_min},{self.y_min} {self.x_min},{self.y_max} " \
            f"{self.x_max},{self.y_max} {self.x_max},{self.y_min}"

    @staticmethod
    def from_xml_coords(coords: str):
        mins, _, maxes, _ = coords.split(' ')
        mins = mins.split(',')
        maxes = maxes.split(',')
        return Rectangle(
            int(mins[0]),
            int(maxes[0]),
            int(mins[1]),
            int(maxes[1]),
        )

    @staticmethod
    def from_json_dict(data_dict):
        return Rectangle(
            *data_dict['coords'],
            data_dict['rect_id'],
            data_dict['content'],
            data_dict['predicted'],
        )

    @staticmethod
    def from_xml_dict(data_dict):
        coords = data_dict['Coords']['@points']
        mins, _, maxes, _ = coords.split(' ')
        mins = mins.split(',')
        maxes = maxes.split(',')
        content = data_dict.get('TextEquiv', None)
        if content:
            content = content['Unicode']
        return Rectangle(
            int(mins[0]),
            int(maxes[0]),
            int(mins[1]),
            int(maxes[1]),
            data_dict.get('@rect_id', None),
            content,
        )

    def to_json_dict(self):
        return {
            'rect_id': self.rect_id,
            'coords': self.coords,
            'content': self.content,
            'predicted': self.predicted,
        }

    def to_xml_dict(self) -> OrderedDict:
        dict_ = OrderedDict({
                '@rect_id': self.rect_id,
                '@type': 'paragraph',
                'Coords': OrderedDict({'@points': self.xml_coords}),
            })
        if self.content:
            dict_['TextEquiv'] = OrderedDict({'Unicode': self.content})
        if self.predicted:
            raise NotImplementedError('Predicted values not available for xml')
        return dict_


def get_rectangle_coords(xs, ys):
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return f"{x_min},{y_min} {x_min},{y_max} {x_max},{y_max} {x_max},{y_min}"


if __name__ == '__main__':
    pass
