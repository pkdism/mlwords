import numpy as np
from _collections import Counter


def mode(target):
    return Counter(target).most_common(1)[0][0]


def mean(target):
    return sum(target)/len(target)


def scale(x):
    x_scale = x
    c = x.shape[1]
    for i in range(c):
        x_scale[:, i] = (x[:, i] - np.mean(x[:, i]))/np.std(x[:, i])
    return x_scale


def euclidean_distance(v1, v2):
    assert len(v1) == len(v2)
    s = 0
    for i in range(len(v1)):
        s += (v1[i] - v2[i]) ** 2
    return s ** 0.5


class KNN:

    def __init__(self, x_train, y_train, x_test, k):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.k = k

    def fit_predict(self, scale_x=True, task="regression"):

        assert self.x_train.shape[1] == self.x_test.shape[1]

        if scale_x:
            x_train = scale(np.vstack((self.x_train, self.x_test)))[0:self.x_train.shape[0], :]
            x_test =  scale(np.vstack((self.x_train, self.x_test)))[self.x_train.shape[0]:, :]
        else:
            x_train = self.x_train
            x_test = self.x_test

        tr = x_train.shape[0]
        te = x_test.shape[0]

        nn = np.zeros((te, tr))

        for test_index in range(te):
            for train_index in range(tr):
                dist = euclidean_distance(x_test[test_index, :], x_train[train_index, :])
                nn[test_index, train_index] = dist

        for test_index in range(te):
            nn[test_index, :] = sorted(range(tr), key=lambda i: nn[test_index, i])

        knn = np.zeros((te, k))
        for i in range(te):
            for j in range(k):
                knn[i, j] = nn[i, j]

        y_test = np.array([])
        np.around(knn)
        for i in range(te):
            ids = list(knn[i, :])
            val = mean(self.y_train[ids]))
            y_test = np.append(y_test, val)

        return y_test
