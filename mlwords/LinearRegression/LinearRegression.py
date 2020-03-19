import numpy as np


class LinearRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rows = self.x.shape[0]
        self.features_count = self.x.shape[1]
        self.theta = np.random.rand(1, self.features_count + 1)
        self.h = np.zeros((self.rows, 1))

    def fit(self, learning_rate=1e-4, max_iter=1000, optimization_algorithm="BatchGradientDescent", batch_size = 1):
        ones = np.ones((self.rows, 1))
        z = np.concatenate((ones, self.x), axis=1)
        cols = z.shape[1]

        if optimization_algorithm == "BatchGradientDescent":
            for iter in range(max_iter):
                self.h = z.dot(self.theta.T)
                dtheta = sum((self.h - self.y.reshape(self.rows, 1)).T.dot(z))
                self.theta = self.theta - learning_rate * dtheta

        if optimization_algorithm == "MiniBatchGradientDescent":
            z_splits = np.array_split(z, self.rows//batch_size, axis=0)
            y_splits = np.array_split(self.y, self.rows // batch_size, axis=0)
            h_splits = np.array_split(self.h, self.rows // batch_size, axis=0)
            n_splits = len(z_splits)
            for iter in range(max_iter):
                for i in range(n_splits):
                    split_rows = z_splits[i].shape[0]
                    h_splits[i] = z_splits[i].dot(self.theta.T)
                    dtheta = sum((h_splits[i] - y_splits[i].reshape(split_rows, 1)).T.dot(z_splits[i]))
                    self.theta = self.theta - learning_rate * dtheta
            self.h = np.concatenate(h_splits)

        h = self.h.reshape(self.rows, 1)
        y = self.y.reshape(self.rows, 1)
        self.training_mse = sum((h - y) ** 2) / self.rows
        self.training_rmse = self.training_mse ** 0.5
        self.training_mae = sum(abs(h - y)) / self.rows
        self.training_mape = sum(abs((h - y) / y)) / self.rows

    def predict(self, x):
        assert x.shape[1] == self.features_count
        ones = np.ones((x.shape[0], 1))
        z = np.concatenate((ones, self.x), axis=1)
        predictions = z.dot(self.theta.T)

        return predictions
