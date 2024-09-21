from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os  

class DataProcessor:
    def __init__(self, window_size, feature_range=(0, 1)):
        """
        Inicializa o processador de dados.

        :param window_size: Tamanho da janela para criar as entradas dos modelos.
        :param feature_range: Intervalo de normalização do MinMaxScaler.
        """
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def create_windows(self, data):
        """
        Cria janelas de dados para entrada em modelos.

        :param data: Array com os dados de preços ou valores.
        :return: Arrays X (entradas) e y (saídas).
        """
        X, y = [], []
        # Se 'dados' for um DataFrame (múltiplas colunas)
        if len(data.shape) == 2:
            # Se a coluna alvo for fornecida, usar essa coluna. Caso contrário, usar a última coluna como padrão
            if coluna_alvo is None:
                coluna_alvo = data.shape[1] - 1  # Última coluna se não especificar
            for i in range(self.window_size, len(data)):
                # Seleciona todas as colunas nas janelas
                X.append(data[i-self.window_size:i])  # Últimos 'window_size' períodos para todas as features
                y.append(data[i, coluna_alvo])  # Valor da coluna alvo no próximo período

        # Se 'dados' for uma única coluna (1D array)
        elif len(data.shape) == 1:
            for i in range(self.window_size, len(data)):
                # Seleciona os últimos 'window_size' períodos
                X.append(data[i-self.window_size:i])
                y.append(data[i])  # Próximo valor da mesma coluna

        # Convertendo para numpy arrays
        X, y = np.array(X), np.array(y)

        # Se o dado de entrada for um DataFrame ou array 2D, precisamos manter o formato 3D
        if len(X.shape) == 2:  # Caso seja 2D, adicionar uma dimensão extra para ser compatível com LSTM
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


        return X_train, X_validation, X_test,y_train, y_validation, y_test

    def normalize(self, X, y):
        """
        Normaliza os dados de treino (X e y).
        
        :param X: Conjunto de entradas (pode ser 2D ou 3D).
        :param y: Conjunto de saídas.
        :return: Dados normalizados.
        """
        X_train_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[2]))
        y_train_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
    
        
        return X_train_scaled, y_train_scaled

    def apply_normalization(self, X, y):
        """
        Aplica a normalização nos dados de entrada (X) e valores alvo (y).
        :param X: Dados de entrada 3D (samples, timesteps, features).
        :param y: Valores alvo 2D (samples, 1).
        :return: Dados de entrada normalizados (X_normalized) e valores alvo normalizados (y_normalized).
        """
        X_normalized = self.scaler.transform(X.reshape(-1, X.shape[2]))
        y_normalized = self.scaler.transform(y.reshape(-1, 1))
        return X_normalized, y_normalized
    
    def reshape_to_original_shape(self, X_scaled, original_shape):
        """
        Redimensiona os dados normalizados de volta para o formato 3D original.
        
        :param X_scaled: Dados normalizados em 2D (samples, timesteps * features).
        :param original_shape: Forma original dos dados em 3D (samples, timesteps, features).
        :return: Dados redimensionados para o formato 3D original.
        """
        # Redimensionar de volta para o formato original (samples, timesteps, features)
        return X_scaled.reshape(original_shape[0], original_shape[1], original_shape[2])

    def inverse_transform(self, y_scaled):
        """
        Desfaz a normalização dos dados.

        :param y_scaled: Dados normalizados.
        :return: Dados desnormalizados.
        """
        return self.scaler.inverse_transform(y_scaled)

    def save_scaler(self, path='scaler/scaler.pkl'):
        """
        Salva o scaler para uso futuro.

        :param path: Caminho para salvar o scaler.
        """
        # Verifica se o diretório existe, se não, cria-o
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório {directory} criado.")

        # Salvar o scaler no caminho especificado
        joblib.dump(self.scaler, path)
        print(f"Scaler salvo em {path}")

    def load_scaler(self, path='scaler/scaler.pkl'):
        """
        Carrega um scaler salvo anteriormente.

        :param path: Caminho do arquivo do scaler salvo.
        """
        self.scaler = joblib.load(path)