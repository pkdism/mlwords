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
