import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean


def downscale_digits(
        data,
        old_shape,
        new_shape,
        cval=0,
):
    label, img = np.hsplit(data, [1])
    factors = old_shape[0] / new_shape[0], old_shape[1] / new_shape[1]
    # todo
    return np.hstack([label, new])


if __name__ == '__main__':
    os.chdir('../../data')
    data = pd.read_csv('labeled.csv', index_col=0)
    data = downscale_digits(data, (50, 50), (10, 10))
    print(data)
