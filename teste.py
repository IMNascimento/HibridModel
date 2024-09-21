import os
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from src.utils.csv_handler import CSVHandler
from src.utils.technical_indicators import TechnicalIndicators


class DataProcessor:
    """Classe responsável por pré-processar os dados, criar janelas e normalizá-los."""
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, file_paths):
        """Carrega os dados de múltiplos CSVs e retorna a coluna 'close'."""
        df = CSVHandler.read_multiple_csvs(file_paths)
        return df['close'].values

    def create_windows(self, data):
        """Cria janelas de dados para LSTM."""
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        return X.reshape((X.shape[0], X.shape[1], 1)), y  # (samples, timesteps, features)

    def split_data(self, X, y, train_size=0.7, val_size=0.15):
        """Divide os dados em treino, validação e teste."""
        train_len = int(len(X) * train_size)
        val_len = int(len(X) * val_size)
        X_train, X_val, X_test = X[:train_len], X[train_len:train_len + val_len], X[train_len + val_len:]
        y_train, y_val, y_test = y[:train_len], y[train_len:train_len + val_len], y[train_len + val_len:]
        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize(self, X_train, y_train):
        """Normaliza os dados de treino e retorna o scaler para aplicação posterior."""
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1))
        return X_train_scaled, y_train_scaled

    def apply_normalization(self, X, y):
        """Aplica a normalização para dados de validação e teste."""
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        y_scaled = self.scaler.transform(y.reshape(-1, 1))
        return X_scaled, y_scaled


class ModelTrainer:
    """Classe responsável por treinar modelos LSTM e de Regressão Linear."""
    
    def __init__(self, batch_size, units, dropout, epochs):
        self.batch_size = batch_size
        self.units = units
        self.dropout = dropout
        self.epochs = epochs

    def train_lstm(self, X_train, y_train, X_val, y_val):
        """Treina o modelo LSTM."""
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
        return model

    def train_linear_regression(self, X_train, y_train):
        """Treina o modelo de Regressão Linear."""
        reg = LinearRegression()
        reg.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        return reg

    def save_model(self, model, description, window_size, directory='models'):
        """Salva o modelo treinado."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        if hasattr(model, 'save'):
            model.save(f'{directory}/lstm_model_{description}.h5')
        else:
            joblib.dump(model, f'{directory}/linear_model_{description}.pkl')


class ResultPlotter:
    """Classe responsável por gerar gráficos para as previsões dos modelos."""
    
    @staticmethod
    def plot_comparison(y_real, y_pred_lstm, y_pred_hybrid, save_path='comparison.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(y_real, color='blue', label='Preço Real')
        plt.plot(y_pred_lstm, color='green', label='Previsão LSTM')
        plt.plot(y_pred_hybrid, color='red', label='Previsão Híbrida')
        plt.title('Comparação entre Preço Real, Previsão LSTM e Previsão Híbrida')
        plt.xlabel('Período')
        plt.ylabel('Preço de Fechamento')
        plt.legend()
        plt.savefig(save_path)
        plt.close()


class ModelPipeline:
    """Pipeline principal que orquestra as operações de treinamento e execução de modelos."""
    
    def __init__(self, window_size, batch_size, units, dropout, epochs):
        self.processor = DataProcessor(window_size)
        self.trainer = ModelTrainer(batch_size, units, dropout, epochs)

    def run(self, data_paths):
        """Executa o pipeline completo: carregamento de dados, treinamento e exibição de resultados."""
        precos_fechamento = self.processor.load_data(data_paths)
        X, y = self.processor.create_windows(precos_fechamento)
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.split_data(X, y)

        X_train_scaled, y_train_scaled = self.processor.normalize(X_train, y_train)
        X_val_scaled, y_val_scaled = self.processor.apply_normalization(X_val, y_val)
        X_test_scaled, y_test_scaled = self.processor.apply_normalization(X_test, y_test)

        # Treinamento de modelos
        lstm_model = self.trainer.train_lstm(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        linear_model = self.trainer.train_linear_regression(X_train_scaled, y_train_scaled)

        # Previsões
        y_pred_lstm = lstm_model.predict(X_test_scaled)
        y_pred_linear = linear_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))

        # Hibridização
        y_pred_hybrid = 0.5 * y_pred_lstm + 0.5 * y_pred_linear

        # Gráfico de comparação
        ResultPlotter.plot_comparison(y_test_scaled, y_pred_lstm, y_pred_hybrid)


def main_menu():
    pipeline = ModelPipeline(window_size=350, batch_size=16, units=70, dropout=0.4, epochs=50)
    data_paths = [
        'output/binance/bitcoin/hourly/BTCUSDT_hourly_2021_2021.csv',
        'output/binance/bitcoin/hourly/BTCUSDT_hourly_2022_2022.csv',
        'output/binance/bitcoin/hourly/BTCUSDT_hourly_2023_2023.csv'
    ]
    
    pipeline.run(data_paths)

    while True:
        print("\nEscolha uma opção:")
        print("1 - Treinar e Hibridizar Modelos")
        print("2 - Sair")

        escolha = input("Digite o número da opção desejada: ")

        if escolha == '1':
            pipeline.run(data_paths)
        elif escolha == '2':
            print("Saindo...")
            break
        else:
            print("Opção inválida, por favor tente novamente.")


if __name__ == '__main__':
    main_menu()