import numpy as np


class LinearRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rows = self.x.shape[0]
        self.features_count = self.x.shape[1]
        self.theta = np.random.rand(1, self.features_count + 1)
        self.h = np.zeros((self.rows, 1))
        self.rmse = np.empty(1)

    def fit(self, learning_rate=1e-4, max_iter=1000):
        ones = np.ones((self.rows, 1))
        z = np.concatenate((ones, self.x), axis=1)
        cols = z.shape[1]

        for iter in range(max_iter):
            self.h = z.dot(self.theta.T)
            dtheta = sum((self.h - self.y.reshape(self.rows, 1)).T.dot(z))
            self.theta = self.theta - learning_rate * dtheta

        self.rmse = sum((self.h.reshape(self.rows, 1) - self.y.reshape(self.rows, 1)) ** 2) / self.rows