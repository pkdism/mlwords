class Metrics:
    def __int__(self):
        pass

    def mse(self, actual, predictions):
        rows = predictions.shape[0]
        h = predictions.reshape(rows, 1)
        y = actual.reshape(rows, 1)

        test_mse = sum((h - y) ** 2) / rows
        return test_mse

    def rmse(self, actual, predictions):
        return self.mse(actual, predictions) ** 0.5

    def mae(self, actual, predictions):
        rows = predictions.shape[0]
        h = predictions.reshape(rows, 1)
        y = actual.reshape(rows, 1)
        test_mae = sum(abs(h - y)) / rows
        return test_mae

    def mape(self, actual, predictions):
        rows = predictions.shape[0]
        h = predictions.reshape(rows, 1)
        y = actual.reshape(rows, 1)
        test_mape = sum(abs((h - y) / y)) / rows
        return test_mape

    def get_regression_metrics(self, actual, predictions):
        test_mse = self.mse(actual, predictions)
        test_rmse = self.rmse(actual, predictions)
        test_mae = self.mae(actual, predictions)
        test_mape = self.mape(actual, predictions)

        return {"MSE": test_mse,
                "RMSE": test_rmse,
                "MAE": test_mae,
                "MAPE": test_mape}

    def confusion_matrix(self, actual, predictions, positive_class=True):
        assert len(actual) == len(predictions)
        actual_binary = np.array([x == positive_class for x in actual])
        predictions_binary = np.array([x == positive_class for x in predictions])
        tp, tn, fp, fn = 0, 0, 0, 0
        l = len(actual_binary)
        for i in range(l):
            tp += actual_binary[i] == predictions_binary[i] and actual_binary[i]==True
            tn += actual_binary[i] == predictions_binary[i] and actual_binary[i]==False
            fp += actual_binary[i] != predictions_binary[i] and actual_binary[i]==False
            fn += actual_binary[i] != predictions_binary[i] and actual_binary[i]==True
        return {"Positive class": positive_class, "TP": tp, "TN": tn, "FP": fp, "FN": fn}

    def precision(self, actual, predictions, positive_class=True):
        cf = self.confusion_matrix(actual=actual, predictions=predictions, positive_class=positive_class)
        return cf["TP"] / (cf["TP"] + cf["FP"])

    def recall(self, actual, predictions, positive_class=True):
        cf = self.confusion_matrix(actual=actual, predictions=predictions, positive_class=positive_class)
        return cf["TP"] / (cf["TP"] + cf["FN"])

    def accuracy(self, actual, predictions, positive_class=True):
        cf = self.confusion_matrix(actual=actual, predictions=predictions, positive_class=positive_class)
        return (cf["TP"] + cf["TN"]) / (cf["TP"] + cf["FP"] + cf["FN"] + cf["TN"])

    def f1(self, actual, predictions, positive_class=True):
        cf = self.confusion_matrix(actual=actual, predictions=predictions, positive_class=positive_class)
        p = self.precision(actual=actual, predictions=predictions, positive_class=positive_class)
        r = self.recall(actual=actual, predictions=predictions, positive_class=positive_class)
        return 2*p*r/(p+r)