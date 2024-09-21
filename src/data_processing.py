from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os  
import pandas as pd

class DataProcessor:
    def __init__(self, window_size, feature_range=(0, 1)):
        """
        Inicializa o processador de dados.

        :param window_size: Tamanho da janela para criar as entradas dos modelos.
        :param feature_range: Intervalo de normalização do MinMaxScaler.
        """
        self.window_size = window_size
        self.scaler_X = MinMaxScaler(feature_range=feature_range)  # Scaler para os dados de entrada (X)
        self.scaler_y = MinMaxScaler(feature_range=feature_range)  # Scaler para a coluna alvo (y)

    def create_windows(self, data, coluna_alvo=None):
        """
        Cria janelas de dados para entrada em modelos.

        :param data: Array com os dados de preços ou valores.
        :return: Arrays X (entradas) e y (saídas).
        """
        X, y = [], []
        # Se 'data' for um DataFrame (várias colunas)
        if isinstance(data, pd.DataFrame):
            # Se a coluna alvo for passada como string (nome da coluna)
            if isinstance(coluna_alvo, str):
                target_col = data[coluna_alvo].values
            else:
                # Usar a última coluna como padrão se 'coluna_alvo' não for especificada
                target_col = data.iloc[:, -1].values
            
            # Percorrer e criar as janelas
            for i in range(self.window_size, len(data)):
                X.append(data.iloc[i-self.window_size:i].values)  # Últimos 'window_size' períodos para todas as features
                y.append(target_col[i])  # Valor da coluna alvo no próximo período

        # Se 'data' for uma única coluna (array ou série)
        elif len(data.shape) == 1:
            for i in range(self.window_size, len(data)):
                # Seleciona os últimos 'window_size' períodos
                X.append(data[i-self.window_size:i])
                y.append(data[i])  # Próximo valor da mesma coluna

        # Convertendo para numpy arrays
        X, y = np.array(X), np.array(y)

        # Se o dado de entrada for um DataFrame ou array 2D, precisamos manter o formato 3D para o LSTM
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)
        
        return X, y

    def split_data(self, X, y, train_size=0.7, validation_size=0.15):
        """
        Divide os dados em conjunto de treino, validação e teste.

        :param X: Conjunto de entradas.
        :param y: Conjunto de saídas.
        :param train_size: Proporção dos dados de treino.
        :param validation_size: Proporção dos dados de validação.
        :return: Conjuntos de treino, validação e teste.
        """
        train_size = int(len(X) * train_size)
        validation_size = int(len(X) * validation_size)

        X_train, X_validation, X_test = X[:train_size], X[train_size:train_size + validation_size], X[train_size + validation_size:]
        y_train, y_validation, y_test = y[:train_size], y[train_size:train_size + validation_size], y[train_size + validation_size:]

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def normalize(self, X, y):
        """
        Normaliza os dados de treino (X e y).
        
        :param X: Conjunto de entradas 3D (samples, timesteps, features).
        :param y: Conjunto de saídas.
        :return: Dados normalizados.
        """
        # Ajusta o scaler_X nos dados de treino e normaliza
        X_train_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[2]))  # Flatten para ajustar o scaler
        y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))  # Normalizar o alvo (y)
        
        return X_train_scaled, y_train_scaled

    def apply_normalization(self, X, y):
        """
        Aplica a normalização nos dados de validação ou teste, ajustada com base nos dados de treino.
        :param X: Dados de entrada 3D (samples, timesteps, features).
        :param y: Valores alvo 2D (samples, 1).
        :return: Dados de entrada normalizados (X_normalized) e valores alvo normalizados (y_normalized).
        """
        try:
            X_normalized = self.scaler_X.transform(X.reshape(-1, X.shape[2]))  # Normalizar com base no ajuste do treino
            y_normalized = self.scaler_y.transform(y.reshape(-1, 1))  # Normalizar os valores de saída
            return X_normalized, y_normalized
        except ValueError as e:
            raise ValueError(f"O MinMaxScaler foi ajustado com {self.scaler_X.n_features_in_} features, mas o X tem {X.shape[2]} features.")
    
    def reshape_to_original_shape(self, X_scaled, original_shape):
        """
        Redimensiona os dados normalizados de volta para o formato 3D original.
        
        :param X_scaled: Dados normalizados em 2D (samples, timesteps * features).
        :param original_shape: Forma original dos dados em 3D (samples, timesteps, features).
        :return: Dados redimensionados para o formato 3D original.
        """
        return X_scaled.reshape(original_shape[0], original_shape[1], original_shape[2])

    def inverse_transform(self, y_scaled):
        """
        Desfaz a normalização dos dados de saída (y).
        """
        return self.scaler_y.inverse_transform(y_scaled)

    def save_scaler(self, path='scaler/scaler.pkl'):
        """
        Salva o scaler ajustado para uso futuro.
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório {directory} criado.")

        # Salvar os scalers separadamente
        joblib.dump(self.scaler_X, os.path.join(directory, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(directory, 'scaler_y.pkl'))
        print(f"Scalers salvos em {directory}")

    def load_scaler(self, path='scaler/scaler.pkl'):
        """
        Carrega os scalers salvos para X e y.
        """
        directory = os.path.dirname(path)
        self.scaler_X = joblib.load(os.path.join(directory, 'scaler_X.pkl'))
        self.scaler_y = joblib.load(os.path.join(directory, 'scaler_y.pkl'))