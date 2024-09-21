from src.utils.csv_handler import CSVHandler
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.metrics_evaluation import ModelEvaluator
from src.time_series_analyzer import TimeSeriesAnalyzer
from src.utils.file_manager import FileManager
from src.plotting import ResultPlotter
from src.utils.logger import Logger
import numpy as np
import random
import tensorflow as tf
import json


# Fixar a semente para garantir reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# Inicializar o Logger com um arquivo de log opcional e nível de logging
Logger.initialize(log_file='logs/treinamento.log', level='INFO')
Logger.info("Iniciando o sistema de treinamento de modelos.")
# Configurations for training LSTM and Regression models


# Função para carregar os parâmetros do JSON
def load_config_from_json(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            Logger.info(f"Configurações carregadas de {config_file}")
            return config
    except FileNotFoundError:
        Logger.error(f"Arquivo de configuração {config_file} não encontrado.")
        raise
    except json.JSONDecodeError as e:
        Logger.error(f"Erro ao carregar o arquivo JSON: {e}")
        raise










def menu():
    # Caminho para o arquivo de configuração
    config_file = 'hiperParams.json'
    
    # Carregar parâmetros do arquivo JSON inicialmente
    config = load_config_from_json(config_file)


    # Pegar os parâmetros carregados
    batch_size = config.get("batch_size", 16)  # Padrão é 16 se não estiver no JSON
    units = config.get("units", 70)
    dropout = config.get("dropout", 0.4)
    epochs = config.get("epochs", 50)
    window_size = config.get("input_window_size", 600)
    data = config.get("datasets", [])
    step_ahead = config.get("output_window_size", 1)
    
    # Carregar os dados
    df = CSVHandler.read_multiple_csvs(data)
    prices = df['close'].values
    #features = ['close', 'high', 'low', 'open']
    #prices = df[features]

    # Initialize the Preprocessor
    processor = DataProcessor(window_size=window_size)


    # Criar janelas
    # EM MULTI FEATURES COLOQUE O NOME DAS COLUNAS QUE DESEJA UTILIZAR COMO ALVO
    X, y = processor.create_windows(prices, steps_ahead=step_ahead)
    # Dividir os dados em treino, validação e teste
    X_train, X_validation, X_test,y_train, y_validation, y_test = processor.split_data(X, y)

    # Normalizar os dados de treino
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)

    # Aplicar normalização nos dados de validação e teste
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)

    # Redimensionar os dados normalizados para o formato original
    X_train_scaled = processor.reshape_to_original_shape(X_train_scaled, X_train.shape)
    X_validation_scaled = processor.reshape_to_original_shape(X_validation_scaled, X_validation.shape)
    X_test_scaled = processor.reshape_to_original_shape(X_test_scaled, X_test.shape)

    # Salvar o scaler para uso posterior
    processor.save_scaler()

    # Initialize the Model Trainer
    model_trainer = ModelTrainer(batch_size, units, dropout, epochs, window_size, step_ahead)


    while True:
        print("\nEscolha uma opção:")
        print("1 - Treinar Regressão Linear")
        print("2 - Treinar LSTM")
        print("3 - Executar modelo LSTM")
        print("4 - Executar modelo Regressão Linear")
        print("5 - Executar modelo híbrido")
        print("6 - Executar Todos os Modelos")
        print("7 - Testar Modelo com Nova Base de Dados (ex: Preços Bitcoin 2024)")
        print("8 - Gerar Matriz de Autocorrelação")
        print("9 - Gerar Correlação entre Indicadores e Preço de Fechamento")
        print("10 - Recarregar Configurações do JSON")
        

        print("11 - Sair")

        escolha = input("Digite o número da opção desejada: ")

        if escolha == '1':
            print("Treinando Regressão Linear...")
            print(f"X_train_scaled shape: {X_train_scaled.shape}")
            print(f"y_train_scaled shape: {y_train_scaled.shape}")
            model_trainer.train_linear_regression(X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled)
        
        elif escolha == '2':
            print("Treinando LSTM...")
            print(f"X_train_scaled shape: {X_train_scaled.shape}")
            print(f"y_train_scaled shape: {y_train_scaled.shape}")
            model_trainer.train_lstm(X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled)

        elif escolha == '3':
            print("Executar modelo LSTM")
            arquivo = FileManager.list_files('models')
            if arquivo is None:
                continue
            
            modelo = model_trainer.loading_model(arquivo)
            y_pred = modelo.predict(X_test_scaled)
            y_pred = processor.inverse_transform(y_pred)
            y_test = processor.inverse_transform(y_test_scaled)
            print(f"Previsões (y_pred): {y_pred[:5]}")
            print(f"y_test shape: {y_test.shape}")
            # Avaliar as métricas do modelo LSTM
            metrics_lstm = ModelEvaluator.evaluate(y_test, y_pred, steps_ahead=step_ahead)
            print("Métricas do LSTM:")
            # Exibir as métricas calculadas
            ModelEvaluator.print_metrics(metrics_lstm)
            ResultPlotter.plot_comparison(y_test, y_pred,title="Modelo LSTM" ,save_path=f'modelo_LSTM_Window_{window_size}_Pre_{step_ahead}.png')

        elif escolha == '4':
            print("Executar modelo Regressão Linear...")
            arquivo = FileManager.list_files('models')
            if arquivo is None:
                continue
            modelo = model_trainer.loading_model(arquivo)
            X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)  # Achatar para regressão linear
            y_pred = modelo.predict(X_test_flat)
            y_pred = processor.inverse_transform(y_pred)
            y_test = processor.inverse_transform(y_test_scaled)
            # Avaliar as métricas do modelo Regressão Linear
            metrics_regressao = ModelEvaluator.evaluate(y_test, y_pred, steps_ahead=step_ahead)
            print("Métricas da Regressão Linear:")
            ModelEvaluator.print_metrics(metrics_regressao)
            ResultPlotter.plot_comparison(y_test, y_pred, title="Modelo de Regressão Linear", save_path=f'modelo_regressao_linear_Window_{window_size}_Pre_{step_ahead}.png')

        elif escolha == '5':
            print("Executar modelo híbrido...")
            
            # Carregar modelo LSTM
            arquivo_lstm = FileManager.list_files('models')
            if arquivo_lstm is None:
                continue
            modelo_lstm = model_trainer.loading_model(arquivo_lstm)

            # Fazer previsões com o LSTM
            y_pred_lstm = modelo_lstm.predict(X_test_scaled)
            y_pred_lstm = processor.inverse_transform(y_pred_lstm)

            # Carregar modelo de Regressão Linear
            arquivo_regressao = FileManager.list_files('models')
            if arquivo_regressao is None:
                continue
            modelo_regressao = model_trainer.loading_model(arquivo_regressao)

            # Achatar os dados de teste para a Regressão Linear
            X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
            
            # Fazer previsões com a Regressão Linear
            y_pred_linear = modelo_regressao.predict(X_test_flat)
            y_pred_linear = processor.inverse_transform(y_pred_linear)

            # Hibridizar os resultados (combinação simples 50% LSTM e 50% Regressão Linear)
            y_pred_hibrido = model_trainer.hybridize_models(y_pred_lstm, y_pred_linear)

            # Recuperar valores reais para plotagem
            y_test = processor.inverse_transform(y_test_scaled)

            # Avaliar as métricas do modelo Híbrido
            metrics_hibrido = ModelEvaluator.evaluate(y_test, y_pred_hibrido, steps_ahead=step_ahead)
            print("\nMétricas do Modelo Híbrido:")
            ModelEvaluator.print_metrics(metrics_hibrido)

            # Plotar comparação Híbrido
            ResultPlotter.plot_comparison(y_test, y_pred_hibrido, title="Modelo Híbrido (LSTM + Regressão Linear)", save_path=f'modelo_hibrido_Window_{window_size}_Pre_{step_ahead}.png')

        elif escolha == '6':
            print("Executar Todos os Modelos...")
            
            # Carregar modelo LSTM
            arquivo_lstm = FileManager.list_files('models')
            if arquivo_lstm is None:
                continue
            modelo_lstm = model_trainer.loading_model(arquivo_lstm)

            # Fazer previsões com o LSTM
            y_pred_lstm = modelo_lstm.predict(X_test_scaled)
            y_pred_lstm = processor.inverse_transform(y_pred_lstm)

            # Carregar modelo de Regressão Linear
            arquivo_regressao = FileManager.list_files('models')
            if arquivo_regressao is None:
                continue
            modelo_regressao = model_trainer.loading_model(arquivo_regressao)

            # Achatar os dados de teste para a Regressão Linear
            X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
            
            # Fazer previsões com a Regressão Linear
            y_pred_linear = modelo_regressao.predict(X_test_flat)
            y_pred_linear = processor.inverse_transform(y_pred_linear)

            # Hibridizar os resultados (combinação simples 50% LSTM e 50% Regressão Linear)
            y_pred_hibrido = model_trainer.hybridize_models(y_pred_lstm, y_pred_linear)

            # Recuperar valores reais para plotagem
            y_test = processor.inverse_transform(y_test_scaled)

            # Avaliar as métricas do modelo Regressão Linear
            metrics_regressao = ModelEvaluator.evaluate(y_test, y_pred_linear, steps_ahead=step_ahead)
            print("Métricas da Regressão Linear:")
            ModelEvaluator.print_metrics(metrics_regressao)

            # Avaliar as métricas do modelo LSTM
            metrics_lstm = ModelEvaluator.evaluate(y_test, y_pred_lstm, steps_ahead=step_ahead)
            print("\nMétricas do LSTM:")
            ModelEvaluator.print_metrics(metrics_lstm)

            # Avaliar as métricas do modelo Híbrido
            metrics_hibrido = ModelEvaluator.evaluate(y_test, y_pred_hibrido, steps_ahead=step_ahead)
            print("\nMétricas do Modelo Híbrido:")
            ModelEvaluator.print_metrics(metrics_hibrido)

            # Plotar comparação LSTM
            ResultPlotter.plot_comparison(y_test, y_pred_lstm, title="Modelo LSTM", save_path=f'modelo_LSTM_Win_{window_size}_Pre_{step_ahead}.png')

            # Plotar comparação Regressão Linear
            ResultPlotter.plot_comparison(y_test, y_pred_linear, title="Modelo de Regressão Linear", save_path=f'modelo_regressao_linear_Win_{window_size}_Pre_{step_ahead}.png')

            # Plotar comparação Híbrido
            ResultPlotter.plot_comparison(y_test, y_pred_hibrido, title="Modelo Híbrido (LSTM + Regressão Linear)", save_path=f'modelo_hibrido_Win_{window_size}_Pre_{step_ahead}.png')

            print("Comparação entre LSTM, Regressão Linear e Híbrido concluída e salva.")

        elif escolha == '7':
            print("Testar Modelo com Nova Base de Dados...")

            # Carregar nova base de dados
            # Listar e selecionar a base de dados, navegando pelas pastas
            caminho_base_dados = FileManager.list_files("data/bitcoin/hourly")
            if not caminho_base_dados:
                return
            
            # Carregar os dados
            df = CSVHandler.read_from_csv(caminho_base_dados)
            prices = df['close'].values

            # Criar janelas
            X, y = processor.create_windows(prices, steps_ahead=step_ahead)

            # Aplicar normalização nos dados de validação e teste
            X_scaled, y_scaled = processor.apply_normalization(X, y)

            # Redimensionar os dados normalizados para o formato original
            X_scaled = processor.reshape_to_original_shape(X_scaled, X.shape)

            # Carregar modelo LSTM
            arquivo_lstm = FileManager.list_files('models')
            if arquivo_lstm is None:
                continue
            modelo_lstm = model_trainer.loading_model(arquivo_lstm)

            # Fazer previsões com o LSTM
            y_pred_lstm = modelo_lstm.predict(X_scaled)
            y_pred_lstm = processor.inverse_transform(y_pred_lstm)

            # Carregar modelo de Regressão Linear
            arquivo_regressao = FileManager.list_files('models')
            if arquivo_regressao is None:
                continue
            modelo_regressao = model_trainer.loading_model(arquivo_regressao)

            # Achatar os dados de teste para a Regressão Linear
            X_scaled_flat = X_scaled.reshape(X_scaled.shape[0], -1)
            
            # Fazer previsões com a Regressão Linear
            y_pred_linear = modelo_regressao.predict(X_scaled_flat)
            y_pred_linear = processor.inverse_transform(y_pred_linear)

            # Hibridizar os resultados (combinação simples 50% LSTM e 50% Regressão Linear)
            y_pred_hibrido = model_trainer.hybridize_models(y_pred_lstm, y_pred_linear)

            # Recuperar valores reais para plotagem
            y_test = processor.inverse_transform(y_scaled)

            # Avaliar as métricas do modelo Regressão Linear
            metrics_regressao = ModelEvaluator.evaluate(y_test, y_pred_linear, steps_ahead=step_ahead)
            print("Métricas da Regressão Linear Nova Base:")
            ModelEvaluator.print_metrics(metrics_regressao)

            # Avaliar as métricas do modelo LSTM
            metrics_lstm = ModelEvaluator.evaluate(y_test, y_pred_lstm, steps_ahead=step_ahead)
            print("\nMétricas do LSTM Nova Base:")
            ModelEvaluator.print_metrics(metrics_lstm)

            # Avaliar as métricas do modelo Híbrido
            metrics_hibrido = ModelEvaluator.evaluate(y_test, y_pred_hibrido, steps_ahead=step_ahead)
            print("\nMétricas do Modelo Híbrido Nova Base:")
            ModelEvaluator.print_metrics(metrics_hibrido)

            # Plotar comparação LSTM
            ResultPlotter.plot_comparison(y_test, y_pred_lstm, title="Modelo LSTM", save_path=f'modelo_LSTM_new_base_pre_{step_ahead}.png')

            # Plotar comparação Regressão Linear
            ResultPlotter.plot_comparison(y_test, y_pred_linear, title="Modelo de Regressão Linear", save_path=f'modelo_regressao_linear_new_base_pre_{step_ahead}.png')

            # Plotar comparação Híbrido
            ResultPlotter.plot_comparison(y_test, y_pred_hibrido, title="Modelo Híbrido (LSTM + Regressão Linear)", save_path=f'modelo_hibrido_new_base_pre_{step_ahead}.png')

            print("Comparação entre LSTM, Regressão Linear e Híbrido em nova base de dados concluída e salva.")

        elif escolha == '8':
            print("Gerar Matriz de Autocorrelação.")
            #lista_window_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
            #lista_window_sizes = list(range(10, 1001, 10))
            lista_window_sizes = list(range(100, 1001, 100))
            #lista_window_sizes = list(range(24, 1001, 24))
            autocorrelacoes = TimeSeriesAnalyzer.calculate_correlation(prices, lista_window_sizes)
            ResultPlotter.plot_autocorrelation(autocorrelacoes)
        
        elif escolha == '9':
            print("Gerar Correlação entre Indicadores e Preço de Fechamento...")
            data_frame= TimeSeriesAnalyzer.add_all_indicators(df)
            TimeSeriesAnalyzer.calculate_and_display_correlation(data_frame)

        elif escolha == '10':
            print("Recarregando Configurações do JSON...")
            # Recarregar os parâmetros do JSON
            config = load_config_from_json(config_file)
            # Atualizar os parâmetros com os novos valores
            batch_size = config.get("batch_size", 16)
            units = config.get("units", 70)
            dropout = config.get("dropout", 0.4)
            epochs = config.get("epochs", 50)
            window_size = config.get("input_window_size", 600)
            step_ahead = config.get("output_window_size", 1)
            datasets = config.get("datasets", [])

            # Recarregar os dados com as novas configurações
            df = CSVHandler.read_multiple_csvs(datasets)
            prices = df['close'].values

            # Recriar o processador e os dados
            processor = DataProcessor(window_size=window_size)
            X, y = processor.create_windows(prices, steps_ahead=step_ahead)
            X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y)
            X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
            X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
            X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
            X_train_scaled = processor.reshape_to_original_shape(X_train_scaled, X_train.shape)
            X_validation_scaled = processor.reshape_to_original_shape(X_validation_scaled, X_validation.shape)
            X_test_scaled = processor.reshape_to_original_shape(X_test_scaled, X_test.shape)

            # Atualizar o Model Trainer com as novas configurações
            model_trainer = ModelTrainer(batch_size, units, dropout, epochs, window_size, step_ahead)
            print("Configurações recarregadas com sucesso!")

        elif escolha == '11':
            print("Saindo...")
            break
        else:
            print("Opção inválida, por favor tente novamente.")

if __name__ == '__main__':
    menu()