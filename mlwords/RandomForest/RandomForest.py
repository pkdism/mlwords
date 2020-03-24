import numpy as np
import mlwords.DecisionTree.DecisionTree as dt


class RandomForest:
    
    def __init__(self, x, y, ntrees=100, max_depth=10, min_terminal_node_size=1):
        self.x = x
        self.y = y
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_terminal_node_size = min_terminal_node_size
        self.rows = x.shape[0]
        self.features_count = x.shape[1]
        self.fit()

    def fit(self):
        trees = [None] * self.ntrees
        for i in range(self.ntrees):
            rows = (self.rows ** 0.5) // 1
            cols = (self.features_count ** 0.5) // 1
            x = self.x[np.random.randint(self.x.shape[0], size=rows), :]
            x = x[:, np.random.randint(x.shape[1], size=cols)]
            y = self.y[np.random.randint(self.y.shape[0], size=rows)]
            tree = dt.DecisionTree(x, y)
            trees[i] = tree.fit()
