import pandas as pd
import numpy as np


class NaiveBayes:

    def __init__(self, x, y):
        """
        :param x: pandas data frame - training data excluding target labels
        :param y: pandas series - binary target label for training data
        """
        assert len(y.value_counts()) <= 2, "y should be binary"
        assert 1 in y, "y should be a pandas series or numpy 1d array with positive class as 1"
        assert isinstance(x, pd.DataFrame), "x should be a pandas dataframe"
        self.x = x
        self.y = y
        self.p1 = []
        self.p0 = []

    def fit(self):
        """
        :return: Dictionary with fitted probabilities for 2 classes using naive bayes algorithm
        """

        x = self.x
        y = self.y
        x.columns = np.arange(x.shape[1])

        x_cat = x
        for i in range(x.shape[1]):
            if x[i].dtype.name in ['int64', 'float64'] and len(x[i].unique()) > 10:
                x_cat[i] = pd.qcut(x[i], q=4)

        x1 = x_cat[y == 1]
        x0 = x_cat[y != 1]

        p1, p0 = [], []
        for i in range(x1.shape[1]):
            v1 = x1[i].value_counts() / len(x1[i])
            p1.append(v1)
            v2 = x0[i].value_counts() / len(x0[i])
            p0.append(v2)

        self.p1 = p1
        self.p0 = p0

    def predict(self, x):
        """
        :param x: test data to predict on fitted naive bayes model
        :return: probabilities for y = 1
        """
        assert x.shape[1] == self.x.shape[1], "x should have same columns as training data"
        x.columns = np.arange(x.shape[1])
        p = np.ones(len(x))
        p1 = self.p1
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                val = x.iloc[row, col]
                if val in p1[col]:
                    p[row] = p[row] * p1[col][val]
        return p
