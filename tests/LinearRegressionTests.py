import numpy as np
import mlwords.LinearRegression.LinearRegression as lm
# import unittest
#
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)
#
#
# if __name__ == '__main__':
#     unittest.main()

# BatchGradientDescent
if False:
    rows = 100
    data_x = np.concatenate((3 * np.random.rand(rows, 1), 4 * np.random.rand(rows, 1), np.random.rand(rows, 1) * (-2)),
                            axis=1)
    data_y = 7 + 5 * data_x[:, 0] - 12 * data_x[:, 1] + 0.5 * data_x[:, 2]

    model = lm.LinearRegression(data_x, data_y)
    model.fit(learning_rate=1e-3, max_iter=1000)
    print(model.theta)
    print(model.training_mae)
    print(model.training_mape)
    print(model.training_mse)
    print(model.training_rmse)

    test_x = np.concatenate((2.5 * np.random.rand(rows, 1), 4.5 * np.random.rand(rows, 1), np.random.rand(rows, 1) * (-1.8)), axis=1)
    print(model.predict(test_x))


# MiniBatchGradientDescent
if False:
    rows = 10000
    data_x = np.concatenate((3 * np.random.rand(rows, 1), 4 * np.random.rand(rows, 1), np.random.rand(rows, 1) * (-2)),
                            axis=1)
    data_y = 7 + 5 * data_x[:, 0] - 12 * data_x[:, 1] + 0.5 * data_x[:, 2]

    model = lm.LinearRegression(data_x, data_y)
    model.fit(learning_rate=1e-5, max_iter=50, optimization_algorithm="MiniBatchGradientDescent", mini_batch_size=500)
    print(model.theta)
    print(model.training_mae)
    print(model.training_mape)
    print(model.training_mse)
    print(model.training_rmse)

    test_x = np.concatenate((2.5 * np.random.rand(rows, 1), 4.5 * np.random.rand(rows, 1), np.random.rand(rows, 1) * (-1.8)), axis=1)
    print(model.predict(test_x))


# StochasticGradientDescent
if False:
    rows = 10000
    data_x = np.concatenate((3 * np.random.rand(rows, 1), 4 * np.random.rand(rows, 1), np.random.rand(rows, 1) * (-2)),
                            axis=1)
    data_y = 7 + 5 * data_x[:, 0] - 12 * data_x[:, 1] + 0.5 * data_x[:, 2]

    model = lm.LinearRegression(data_x, data_y)
    model.fit(learning_rate=1e-5, max_iter=1000, optimization_algorithm="StochasticGradientDescent", sgd_rows=1000)
    print(model.theta)
    print(model.training_mae)
    print(model.training_mape)
    print(model.training_mse)
    print(model.training_rmse)

    test_x = np.concatenate((2.5 * np.random.rand(rows, 1), 4.5 * np.random.rand(rows, 1), np.random.rand(rows, 1) * (-1.8)), axis=1)
    print(model.predict(test_x))