# import unittest
import numpy as np
import mlwords.DecisionTree.DecisionTree as dt


# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)


# if __name__ == '__main__':
#     unittest.main()

data_x = np.array([[10, 200, 100, 12, 50, 40, 90, 10, 10, 80],
                   [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
                   [10, 11, 11, 10, 12, 9, 11, 12, 9, 10]]).T
data_y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0])

model = dt.DecisionTree(data_x, data_y)
tree = model.fit()
print(tree)


# test_x = np.array([[11, 180, 110, 12], [1, 0, 0, 0], [10, 11, 10, 10]]).T
# print(model.predict(test_x))