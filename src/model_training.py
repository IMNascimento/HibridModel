from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import os
import joblib
from src.utils.logger import Logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
from src.plotting import ResultPlotter



class ModelTrainer:
    def __init__(self, batch_size=None, units=None, dropout=None, epochs=None, window_size=None, steps_ahead=1):
        self.batch_size = batch_size
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.window_size = window_size
        self.steps_ahead = steps_ahead

    def train_lstm(self, X_train, y_train, X_val, y_val):
        """
        Treina o modelo LSTM.
        """
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=self.steps_ahead))  # 'steps_ahead' saídas

        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
        self.save_model(model, descricao=f'lstm_{self.steps_ahead}_predition')
        return model

    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """
        Treina o modelo de Regressão Linear para prever uma janela completa (steps_ahead) de valores.
        """
        try:
            # Achatar os dados de treino e validação para serem compatíveis com a regressão linear
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)

            # Se steps_ahead > 1, mantemos todos os valores de y_train para prever múltiplos steps à frente
            if len(y_train.shape) > 1 and self.steps_ahead > 1:
                y_train_flat = y_train  # Usamos todos os steps para treino
                y_val_flat = y_val  # Usamos todos os steps para validação
            else:
                y_train_flat = y_train.reshape(-1) if len(y_train.shape) > 1 else y_train
                y_val_flat = y_val.reshape(-1) if len(y_val.shape) > 1 else y_val

            # Instanciar e treinar o modelo de Regressão Linear
            model = LinearRegression()
            model.fit(X_train_flat, y_train_flat)

            # Salvar o modelo de Regressão Linear
            self.save_model(model, descricao=f'regressao_linear_{self.steps_ahead}_predition')

            return model

        except Exception as e:
            Logger.error(f"Erro ao treinar a Regressão Linear: {e}")
            return None
        
    def train_lasso_regression(self, X_train, y_train, X_val, y_val, alpha=0.1):
        """
        Treina o modelo de Regressão Lasso para prever uma janela completa (steps_ahead) de valores.
        """
        try:
            # Achatar os dados de treino e validação para serem compatíveis com o Lasso
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)

            # Verificar a forma de y_train e y_val para steps_ahead
            y_train_flat = y_train.reshape(-1) if len(y_train.shape) > 1 else y_train
            y_val_flat = y_val.reshape(-1) if len(y_val.shape) > 1 else y_val

            # Instanciar e treinar o modelo de Regressão Lasso
            model = Lasso(alpha=alpha)
            model.fit(X_train_flat, y_train_flat)

            # Salvar o modelo de Regressão Lasso
            self.save_model(model, descricao=f'lasso_regression_{self.steps_ahead}_predition_alpha_{alpha}')

            return model

        except Exception as e:
            Logger.error(f"Erro ao treinar a Regressão Lasso: {e}")
            return None

    def train_ridge_regression(self, X_train, y_train, X_val, y_val, alpha=1.0):
        """
        Treina o modelo de Regressão Ridge para prever uma janela completa (steps_ahead) de valores.
        """
        try:
            # Achatar os dados de treino e validação para serem compatíveis com o Ridge
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)

            # Verificar a forma de y_train e y_val para steps_ahead
            y_train_flat = y_train.reshape(-1) if len(y_train.shape) > 1 else y_train
            y_val_flat = y_val.reshape(-1) if len(y_val.shape) > 1 else y_val

            # Instanciar e treinar o modelo de Regressão Ridge
            model = Ridge(alpha=alpha)
            model.fit(X_train_flat, y_train_flat)

            # Salvar o modelo de Regressão Ridge
            self.save_model(model, descricao=f'ridge_regression_{self.steps_ahead}_predition_alpha_{alpha}')

            return model

        except Exception as e:
            Logger.error(f"Erro ao treinar a Regressão Ridge: {e}")
            return None
    
    def save_model(self, model, pasta='models', descricao='')->None:
        """
        Salva o modelo treinado em um diretório específico.
        
        :param model: O modelo treinado (pode ser Keras ou scikit-learn).
        :param batch_size: Tamanho do batch usado no treinamento.
        :param units: Número de unidades no LSTM.
        :param dropout: Taxa de dropout usada no LSTM.
        :param window_size: Tamanho da janela de entrada.
        :param pasta: Caminho para o diretório onde o modelo será salvo (deve ser string).
        :param descricao: Descrição adicional para o nome do arquivo.
        """
                # Verifique se o caminho fornecido é uma string válida
        if not isinstance(pasta, str):
            raise TypeError(f"O caminho 'pasta' deve ser uma string, mas '{type(pasta).__name__}' foi fornecido.")

        # Se o diretório não existir, crie-o
        if not os.path.exists(pasta):
            os.makedirs(pasta)
            Logger.info(f'Diretório {pasta} criado para salvar o modelo.')

        # Criar o nome do arquivo para o modelo
        if hasattr(model, 'save'):
            # Se for um modelo Keras (LSTM)
            caminho_modelo = os.path.join(pasta, f'modelo_{descricao}_batch_{self.batch_size}_units_{self.units}_dropout_{self.dropout}_window_{self.window_size}.h5')
            model.save(caminho_modelo)
            Logger.info(f'Modelo Keras salvo em: {caminho_modelo}')
        else:
            # Se for um modelo scikit-learn (como Regressão Linear)
            caminho_modelo = os.path.join(pasta, f'modelo_{descricao}_window_{self.window_size}.pkl')
            joblib.dump(model, caminho_modelo)
            Logger.info(f'Modelo scikit-learn salvo em: {caminho_modelo}')

    
    def loading_model(self, model_path):
        """
        Carrega um modelo salvo (pode ser um modelo de Regressão Linear ou um modelo Keras).
        
        :param model_path: Caminho do modelo salvo.
        :return: O modelo carregado.
        """
        # Verifica se o arquivo existe
        if not os.path.exists(model_path):
            Logger.error(f"O modelo no caminho {model_path} não foi encontrado.")
            raise FileNotFoundError(f"O modelo no caminho {model_path} não foi encontrado.")
        
        
        # Verifica a extensão do arquivo para decidir como carregar o modelo
        if model_path.endswith('.pkl'):
            # Para modelos salvos com joblib (ex.: scikit-learn como Regressão Linear)
            model = joblib.load(model_path)
            Logger.info(f"Modelo carregado de {model_path} com joblib.")
            
        
        elif model_path.endswith('.h5') or model_path.endswith('.hdf5'):
            # Para modelos Keras salvos com .h5
            model = load_model(model_path)
            Logger.info(f"Modelo Keras carregado de {model_path} com keras.")
        
        else:
            Logger.error("Extensão de arquivo não suportada. Use '.pkl' para modelos scikit-learn ou '.h5' para modelos Keras.")
            raise ValueError("Extensão de arquivo não suportada. Use '.pkl' para modelos scikit-learn ou '.h5' para modelos Keras.")
        
        return model

    
        

    def hybridize_models(self, y_pred_lstm, y_pred_linear):
        """
        Combina as previsões do LSTM e da Regressão Linear de forma híbrida.

        :param y_pred_lstm: Previsões do modelo LSTM.
        :param y_pred_linear: Previsões do modelo de Regressão Linear.
        :return: Previsões híbridas (média ponderada).
        """
        try:
            Logger.info("Hibridizando as previsões dos modelos LSTM e Regressão Linear...")
            y_pred_hybrid = 0.5 * y_pred_lstm + 0.5 * y_pred_linear
            Logger.info("Hibridização concluída com sucesso.")
            return y_pred_hybrid
        except Exception as e:
            Logger.error(f"Erro ao hibridizar os modelos: {e}")
            return None
        
    
    