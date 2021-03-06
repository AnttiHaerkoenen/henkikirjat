import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean

from src.tools.digits import view_digit


def downscale_digits(
        data: pd.DataFrame,
        old_shape,
        new_shape,
        cval=0,
):
    label, img = data['label'], data.drop(columns=['label'])
    factors = old_shape[0] // new_shape[0], old_shape[1] // new_shape[1]
    new = []
    for row in img.itertuples(index=False):
        arr = np.array(row).reshape(old_shape)
        arr = downscale_local_mean(arr, factors=factors, cval=cval)
        new.append(arr.ravel())
    new = pd.DataFrame(new)
    concat = pd.concat([label, new], axis=1)
    return concat


if __name__ == '__main__':
    os.chdir('../../data/train/1900')
    input_ = 'labeled_1900.csv'
    output_ = 'downscaled_1900_10.csv'
    data = pd.read_csv(input_, index_col=0)
    data = downscale_digits(data, (40, 40), (10, 10))
    data.to_csv(output_)
    data = pd.read_csv(output_, index_col=0)
    data = data[pd.notna(data['label'])]
    for idx, row in data.tail(50).iterrows():
        view_digit(row, (10, 10))
