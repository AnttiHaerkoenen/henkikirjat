from collections import OrderedDict
from dataclasses import dataclass
from typing import Union


@dataclass
class Rectangle:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    id: Union[str, None] = None
    content: Union[str, None] = None
    predicted: Union[str, None] = None

    @property
    def h(self):
        return self.y_max - self.y_min

    @property
    def w(self):
        return self.x_max - self.x_min

    @property
    def pil_box(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def np_slice(self):
        return slice(self.y_min, self.y_max), slice(self.x_min, self.x_max)

    @property
    def coords(self) -> str:
        return f"{self.x_min},{self.y_min} {self.x_min},{self.y_max} " \
            f"{self.x_max},{self.y_max} {self.x_max},{self.y_min}"

    @staticmethod
    def from_coords(coords: str):
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
    def from_dict(data_dict: dict):
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
            data_dict.get('@id', None),
            content,
        )

    def to_dict(self) -> OrderedDict:
        dict_ = OrderedDict({
                '@id': self.id,
                '@type': 'paragraph',
                'Coords': OrderedDict({'@points': self.coords}),
            })
        if self.content:
            dict_['TextEquiv'] = OrderedDict({'Unicode': self.content})
        if self.predicted:
            dict_['Predicted'] = OrderedDict({'Unicode': self.predicted})
        return dict_


def get_rectangle_coords(xs, ys):
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return f"{x_min},{y_min} {x_min},{y_max} {x_max},{y_max} {x_max},{y_min}"


if __name__ == '__main__':
    pass
