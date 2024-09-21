import pandas as pd
from src.plotting import ResultPlotter  
from src.utils.logger import Logger
from src.utils.technical_indicators import TechnicalIndicators 
import numpy as np 

class TimeSeriesAnalyzer:
    """
    Classe para análise de séries temporais, incluindo cálculos de autocorrelação e outros métodos de análise.
    """

    @staticmethod
    def calculate_correlation(precos_fechamento, lista_window_sizes):
        autocorrelacoes = {}
        for window_size in lista_window_sizes:
            try:
                Logger.info(f"Calculando autocorrelação para window_size = {window_size}...")

                # Seleciona os últimos 'window_size' valores
                dados_janela = precos_fechamento[-window_size:]

                # Calcular a autocorrelacao
                autocorrelacao = pd.Series(dados_janela).autocorr(lag=1)
                autocorrelacoes[window_size] = autocorrelacao

                Logger.info(f"Autocorrelação para window_size {window_size}: {autocorrelacao}")
            except Exception as e:
                Logger.error(f"Erro ao calcular autocorrelação para window_size {window_size}: {e}")
        return autocorrelacoes


    @staticmethod
    def add_all_indicators(df):
        """
        Adiciona indicadores técnicos ao DataFrame.
        
        :param df: DataFrame contendo os dados de preços.
        :return: DataFrame com indicadores técnicos adicionados.
        """
        try:
            Logger.info("Adicionando indicadores técnicos ao DataFrame...")
        
            # Adicionando SMA
            df['SMA_20'] = TechnicalIndicators.sma(df['close'], period=20)
            df['SMA_50'] = TechnicalIndicators.sma(df['close'], period=50)
            df['SMA_100'] = TechnicalIndicators.sma(df['close'], period=100)
            df['SMA_200'] = TechnicalIndicators.sma(df['close'], period=200)

            # Adicionando EMA
            df['EMA_20'] = TechnicalIndicators.ema(df['close'], period=20)
            df['EMA_50'] = TechnicalIndicators.ema(df['close'], period=50)
            df['EMA_100'] = TechnicalIndicators.ema(df['close'], period=100)
            df['EMA_200'] = TechnicalIndicators.ema(df['close'], period=200)

            # Adicionando RSI
            df['RSI_14'] = TechnicalIndicators.rsi(df['close'], period=14)

            # Adicionando MACD (12, 26, 9 é a configuração padrão para MACD)
            macd_df = TechnicalIndicators.macd(df['close'])
            df = pd.concat([df, macd_df], axis=1)# Adicionando MACD, Signal e Histograma ao DataFrame

            # Adicionando Bandas de Bollinger
            bollinger_df = TechnicalIndicators.bollinger_bands(df['close'], period=20)
            df = pd.concat([df, bollinger_df], axis=1)

            # Adicionando Oscilador Estocástico
            df['Stochastic_K'] = TechnicalIndicators.stochastic_oscillator(df, period=14)

            # Adicionando ADX
            df['ADX_14'] = TechnicalIndicators.adx(df, period=14)

            # Adicionando ATR
            df['ATR_14'] = TechnicalIndicators.atr(df, period=14)

            # Adicionando Envelopes
            envelopes_df = TechnicalIndicators.envelope(df['close'], period=20, percent=3.0)# Envelopes de Preço
            df = pd.concat([df, envelopes_df], axis=1)# Adicionando os Envelopes ao DataFrame

            # Adicionando Ichimoku Kinko Hyo
            ichimoku_df = TechnicalIndicators.ichimoku_kinko_hyo(df)
            df = pd.concat([df, ichimoku_df], axis=1)

            # Adicionando VWAP
            df['VWAP'] = TechnicalIndicators.vwap(df['close'], df['volume'])

            # Adicionando Pivot Points
            pivot_df = TechnicalIndicators.pivot_points(df['high'], df['low'], df['close'])
            df = pd.concat([df, pivot_df], axis=1)

            # Adicionando Fibonacci Retracement
            fibonacci_df = TechnicalIndicators.fibonacci_retracement(df['close'])
            df = pd.concat([df, fibonacci_df], axis=1)

            Logger.info("Indicadores técnicos adicionados com sucesso.")
            
            return df
        
        except Exception as e:
            Logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
            return df

    @staticmethod
    def calculate_and_display_correlation(df):
        """
        Calcula e exibe a correlação entre indicadores técnicos e o preço de fechamento.

        :param df: DataFrame com indicadores técnicos e preços.
        """
        try:
            Logger.info("Calculando correlação entre indicadores técnicos e preço de fechamento...")
            
            # Selecionar apenas colunas numéricas
            df_numerico = df.select_dtypes(include=[np.number])

            # Calcular a correlação com a coluna 'close'
            if 'close' in df_numerico.columns:
                correlacoes = df_numerico.corr()['close'].drop('close')
                Logger.info(f"Correlação calculada:\n{correlacoes}")
            else:
                Logger.warning("A coluna 'close' não está presente no DataFrame ou não é numérica.")
                return

            # Usar o ResultPlotter para plotar e salvar a correlação
            ResultPlotter.plot_correlation(correlacoes, 'Correlação entre Indicadores Técnicos e Preço de Fechamento', save_path='correlacao_indicadores.png')
        
        except Exception as e:
            Logger.error(f"Erro ao calcular e exibir a correlação: {e}")