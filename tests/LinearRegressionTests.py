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


rows = 10000
data_x = np.concatenate((3 * np.random.rand(rows, 1), 4 * np.random.rand(rows, 1), np.random.rand(rows, 1) * (-2)), axis=1)
data_y = 7 + 5 * data_x[:, 0] - 12 * data_x[:, 1] + 0.5 * data_x[:, 2]

model = LinearRegression(data_x, data_y)
model.fit(learning_rate=1e-5, max_iter=1000)

print(model.theta, model.rmse)


