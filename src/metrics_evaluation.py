from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Calcula as métricas de avaliação.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'RMSE': rmse
        }

    @staticmethod
    def print_metrics(metrics):
        """
        Imprime as métricas de avaliação.
        """
        for key, value in metrics.items():
            print(f"{key}: {value}")