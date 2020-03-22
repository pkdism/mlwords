import numpy as np
from math import log2


class DecisionTree:

    def __int__(self, x, y):
        self.x = x
        self.y = y
        self.rows = self.x.shape[0]
        self.features_count = self.x.shape[1]

    def gini(self, feature, classes):
        n = len(classes)
        p1 = 0
        if n != 0:
            p1 = sum(classes) / n
        p0 = 1 - p1
        return p0 * p0 + p1 * p1

    def gini_impurity(self, feature, classes):
        return 1 - self.gini(feature=feature, classes=classes)

    def chi_square(self):
        pass

    def information_gain(self):
        pass

    def variance(self):
        pass

    def find_best_split(self, x, y, split_criterion="gini_impurity"):
        split_values = np.ones(self.features_count) * np.inf
        split_points = np.ones(self.features_count) * np.inf
        for i in range(self.features_count):
            low, high = min(x[:, i]), max(x[:, i])
            vals = np.linspace(low, high, 100)
            for v in vals:
                d1 = x[np.where(x[:, i] < v), i]
                d2 = x[np.where(x[:, i] >= v), i]
                y1 = y[np.where(x[:, i] < v)]
                y2 = y[np.where(x[:, i] >= v)]
                if split_criterion == "gini_impurity":
                    gini_part1 = self.gini_impurity(d1, y1)
                    gini_part2 = self.gini_impurity(d2, y2)
                    n1, n2 = len(y1), len(y2)
                    gv = (n1 * gini_part1 + n2 * gini_part2) / (n1 + n2)
                    if gv < split_values[i]:
                        split_values[i] = gv
                        split_points[i] = v

        best_value = np.inf
        feature_index, feature_value = 0, 0
        for i in range(self.features_count):
            if split_values[i] < best_value:
                feature_index = i
                feature_value = split_points[i]
        return {"feature_index": feature_index, "feature_split_point": feature_value}

    def build_tree(self,
                   x,
                   y,
                   min_terminal_node_size = 1,
                   min_nodes_for_split=2,
                   max_depth=100,
                   cur_depth=0,
                   split_criterion="gini_impurity"):

        node_size = len(y)
        if node_size <= min_nodes_for_split or cur_depth > max_depth:
            return np.nan

        split_feature = self.find_best_split(x=x, y=y, split_criterion=split_criterion)

        d1 = x[np.where(x[:, split_feature["feature_index"]] < split_feature["feature_split_point"]), :]
        d2 = x[np.where(x[:, split_feature["feature_index"]] >= split_feature["feature_split_point"]), :]
        y1 = y[np.where(x[:, split_feature["feature_index"]] < split_feature["feature_split_point"])]
        y2 = y[np.where(x[:, split_feature["feature_index"]] < split_feature["feature_split_point"])]

        return {"feature_index": split_feature["feature_index"],
                "feature_split_point": split_feature["feature_split_point"],
                "yes": self.build_tree(x=d2,
                                       y=y2,
                                       min_terminal_node_size=min_terminal_node_size,
                                       min_nodes_for_split=min_nodes_for_split,
                                       max_depth=max_depth,
                                       cur_depth=cur_depth+1,
                                       split_criterion=split_criterion),
                "no": self.build_tree(x=d1,
                                       y=y1,
                                       min_terminal_node_size=min_terminal_node_size,
                                       min_nodes_for_split=min_nodes_for_split,
                                       max_depth=max_depth,
                                       cur_depth=cur_depth+1,
                                       split_criterion=split_criterion)}

    def fit(self,
            task="binary_classification",
            min_terminal_node_size=1,
            min_nodes_for_split=2,
            max_depth=100,
            split_criterion="gini_impurity"):
        depth = 0
        if task == "binary_classification"
        tree = self.build_tree(x=self.x,
                               y=self.y,
                               min_terminal_node_size=min_terminal_node_size,
                               min_nodes_for_split=min_nodes_for_split,
                               max_depth=max_depth,
                               cur_depth=depth,
                               split_criterion=split_criterion)
        return tree
