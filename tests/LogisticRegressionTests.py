import numpy as np
import mlwords.LogisticRegression.LogisticRegression as glm
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
if True:
    data_x = np.array([[10, 200, 100, 12], [1, 0, 0, 1], [10, 11, 11, 10]]).T
    data_y = np.array([1, 0, 0, 1])

    model = glm.LogisticRegression(data_x, data_y)
    model.fit(learning_rate=1e-3, max_iter=1000)
    print(model.theta)

    test_x = np.array([[11, 180, 110, 12], [1, 0, 0, 0], [10, 11, 10, 10]]).T
    print(model.predict(test_x))


# MiniBatchGradientDescent
if True:
    data_x = np.array([[10, 200, 100, 12], [1, 0, 0, 1], [10, 11, 11, 10]]).T
    data_y = np.array([1, 0, 0, 1])

    model = glm.LogisticRegression(data_x, data_y)
    model.fit(learning_rate=1e-3, max_iter=1000, optimization_algorithm="MiniBatchGradientDescent", mini_batch_size=2)
    print(model.theta)

    test_x = np.array([[11, 180, 110, 12], [1, 0, 0, 0], [10, 11, 10, 10]]).T
    print(model.predict(test_x))


# StochasticGradientDescent
if True:
    data_x = np.array([[10, 200, 100, 12], [1, 0, 0, 1], [10, 11, 11, 10]]).T
    data_y = np.array([1, 0, 0, 1])

    model = glm.LogisticRegression(data_x, data_y)
    model.fit(learning_rate=1e-3, max_iter=1000, optimization_algorithm="StochasticGradientDescent", sgd_rows=2)
    print(model.theta)

    test_x = np.array([[11, 180, 110, 12], [1, 0, 0, 0], [10, 11, 10, 10]]).T
    print(model.predict(test_x))