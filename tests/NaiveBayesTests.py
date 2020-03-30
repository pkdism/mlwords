# import unittest
import numpy as np
import pandas as pd
import mlwords.NaiveBayes.NaiveBayes as nb

# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)


# if __name__ == '__main__':
#     unittest.main()


if True:
    data_x = pd.DataFrame({'A': 1.,
                           'B': pd.Timestamp('20130102'),
                           'C': pd.Series([1, 2, 3, 2]),
                           'D': np.array([3] * 4),
                           'E': pd.Categorical(['test', 'train', 'test', 'train']),
                           'F': pd.Categorical(['train', 'train', 'train', 'test'])})
    data_y = pd.Series(np.array([1, 0, 0, 1]))

    model = nb.NaiveBayes(data_x, data_y)
    model.fit()

    test_x = pd.DataFrame({'A': 1.,
                           'B': pd.Timestamp('20130102'),
                           'C': pd.Series([1, 2]),
                           'D': np.array([3] * 2),
                           'E': pd.Categorical(['test', 'train']),
                           'F': pd.Categorical(['train', 'train'])})
    print(model.predict(test_x))