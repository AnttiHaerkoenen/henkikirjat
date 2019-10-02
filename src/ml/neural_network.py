import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.ml.feature_extraction import downscale_digits


if __name__ == '__main__':
    split = 10000
    os.chdir('../../data')
    data = pd.read_csv('downscaled_10_10.csv')
    data = data[pd.notna(data['label'])]

    X = data.drop(columns=['label'])
    y = data['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=2000,
    )
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    errors = confusion_matrix(y_test, y_predict)
    report = classification_report(y_test, y_predict)
    print(report)
    print(errors)
