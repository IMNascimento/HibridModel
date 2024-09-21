from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred, steps_ahead=1):
        """
        Calcula as métricas de avaliação para múltiplos passos à frente (steps_ahead).
        
        :param y_true: Valores reais.
        :param y_pred: Previsões do modelo.
        :param steps_ahead: Quantidade de passos à frente previstos.
        :return: Dicionário com as métricas para cada step à frente.
        """
        metrics = {}

        if len(y_true.shape) == 1 or steps_ahead == 1:
            # Tratar como unidimensional
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            metrics["Step 1"] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'RMSE': rmse
            }

        elif len(y_true.shape) == 2 and y_true.shape[1] == steps_ahead:
            # Métricas para cada step à frente
            for step in range(steps_ahead):
                mse = mean_squared_error(y_true[:, step], y_pred[:, step])
                mae = mean_absolute_error(y_true[:, step], y_pred[:, step])
                r2 = r2_score(y_true[:, step], y_pred[:, step])
                mape = mean_absolute_percentage_error(y_true[:, step], y_pred[:, step])
                rmse = np.sqrt(mse)

                metrics[f"Step {step + 1}"] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape,
                    'RMSE': rmse
                }
        else:
            raise ValueError(f"y_true e y_pred têm dimensões incompatíveis para {steps_ahead} passos à frente.")

        return metrics

    @staticmethod
    def print_metrics(metrics):
        """
        Imprime as métricas de avaliação.
        """
        for step, step_metrics in metrics.items():
            print(f"Métricas para {step}:")
            for key, value in step_metrics.items():
                print(f"{key}: {value}")
            print("-" * 40)