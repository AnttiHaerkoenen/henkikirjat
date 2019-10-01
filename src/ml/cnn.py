import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf

from src.ml.feature_extraction import downscale_digits


if __name__ == '__main__':
    split = 10000
    os.chdir('../../data')
    data = pd.read_csv('downscaled_10_10.csv')
    label = data['label'].values
    y_train, y_test = label[:split], label[split:]
    x_train, x_test = np.vsplit(data.drop['label'].values, split)
    print(y_test)
    print(x_test)
