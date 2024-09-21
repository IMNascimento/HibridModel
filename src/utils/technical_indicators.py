import pandas as pd

class TechnicalIndicators:
    """
    Classe que implementa métodos estáticos para calcular diversos indicadores técnicos.
    """

    @staticmethod
    def sma(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula a Média Móvel Simples (SMA).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 14.
        :return: Série contendo os valores da SMA.
        """
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula a Média Móvel Exponencial (EMA).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da EMA. Padrão é 14.
        :return: Série contendo os valores da EMA.
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def envelopes(series: pd.Series, period: int = 14, percent: float = 3.0) -> pd.DataFrame:
        """
        Calcula Envelopes de preço ao redor da SMA.
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 14.
        :param percent: Percentual para calcular os envelopes superior e inferior. Padrão é 3%.
        :return: DataFrame com colunas de upper e lower envelopes.
        """
        sma = TechnicalIndicators.sma(series, period)
        upper_envelope = sma * (1 + percent / 100)
        lower_envelope = sma * (1 - percent / 100)
        return pd.DataFrame({
            f'Upper_Envelope_{period}_{percent}%': upper_envelope,
            f'Lower_Envelope_{period}_{percent}%': lower_envelope
        })

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula o Índice de Força Relativa (RSI).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo do RSI. Padrão é 14.
        :return: Série contendo os valores do RSI.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """
        Calcula o MACD (Moving Average Convergence Divergence).
        
        :param series: Série temporal dos preços.
        :param fastperiod: Período rápido para a EMA. Padrão é 12.
        :param slowperiod: Período lento para a EMA. Padrão é 26.
        :param signalperiod: Período do sinal do MACD. Padrão é 9.
        :return: DataFrame contendo MACD, Signal e Histograma.
        """
        fast_ema = TechnicalIndicators.ema(series, period=fastperiod)
        slow_ema = TechnicalIndicators.ema(series, period=slowperiod)
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        hist = macd - signal

        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Hist': hist
        })
    
    @staticmethod
    def bollinger_bands(prices, period=20, num_std=2):
        """
        Calcula as Bandas de Bollinger.
        :param prices: Preços de fechamento.
        :param period: Período da média móvel.
        :param num_std: Número de desvios padrão.
        :return: Média, Banda Superior, Banda Inferior.
        """
        sma = TechnicalIndicators.sma(prices, period)
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return pd.DataFrame({'SMA': sma, 'Upper Band': upper_band, 'Lower Band': lower_band})

    @staticmethod
    def stochastic_oscillator(prices, period=14):
        """
        Calcula o Oscilador Estocástico.
        :param prices: DataFrame com as colunas 'high', 'low', 'close'.
        :param period: Período para cálculo.
        :return: Oscilador estocástico calculado.
        """
        lowest_low = prices['low'].rolling(window=period).min()
        highest_high = prices['high'].rolling(window=period).max()
        k_percent = (prices['close'] - lowest_low) / (highest_high - lowest_low) * 100
        return k_percent

    @staticmethod
    def adx(prices, period=14):
        """
        Calcula o Índice Direcional Médio (ADX).
        :param prices: DataFrame com as colunas 'high', 'low', 'close'.
        :param period: Período para cálculo.
        :return: ADX calculado.
        """
        high = prices['high']
        low = prices['low']
        close = prices['close']

        plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0)
        minus_dm = low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def atr(prices, period=14):
        """
        Calcula o Average True Range (ATR).
        :param prices: DataFrame com as colunas 'high', 'low', 'close'.
        :param period: Período para cálculo.
        :return: ATR calculado.
        """
        high = prices['high']
        low = prices['low']
        close = prices['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def ichimoku_kinko_hyo(prices, short_period=9, medium_period=26, long_period=52):
        """
        Calcula o Ichimoku Kinko Hyo.
        :param prices: DataFrame com as colunas 'high', 'low', 'close'.
        :param short_period: Período curto (Tankan-sen).
        :param medium_period: Período médio (Kijun-sen).
        :param long_period: Período longo (Senkou Span).
        :return: Um DataFrame com as cinco linhas do Ichimoku.
        """
        high = prices['high']
        low = prices['low']
        close = prices['close']

        tankan_sen = (high.rolling(window=short_period).max() + low.rolling(window=short_period).min()) / 2
        kijun_sen = (high.rolling(window=medium_period).max() + low.rolling(window=medium_period).min()) / 2
        senkou_span_a = ((tankan_sen + kijun_sen) / 2).shift(medium_period)
        senkou_span_b = ((high.rolling(window=long_period).max() + low.rolling(window=long_period).min()) / 2).shift(medium_period)
        chikou_span = close.shift(-medium_period)

        return pd.DataFrame({
            'Tankan-sen': tankan_sen,
            'Kijun-sen': kijun_sen,
            'Senkou Span A': senkou_span_a,
            'Senkou Span B': senkou_span_b,
            'Chikou Span': chikou_span
        })

    @staticmethod
    def vwap(prices, volume, period=None):
        """
        Calcula o Volume Weighted Average Price (VWAP).
        :param prices: Preços de fechamento.
        :param volume: Volume de negociação.
        :param period: Período para calcular o VWAP.
        :return: VWAP calculado.
        """
        cum_vol_price = (prices * volume).cumsum()
        cum_volume = volume.cumsum()
        vwap = cum_vol_price / cum_volume
        if period:
            return vwap.rolling(window=period).mean()
        return vwap

    @staticmethod
    def pivot_points(high, low, close):
        """
        Calcula os Pivot Points.
        :param high: Preços máximos.
        :param low: Preços mínimos.
        :param close: Preço de fechamento.
        :return: Um DataFrame com os Pivot Points.
        """
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return pd.DataFrame({
            'Pivot': pivot,
            'R1': r1,
            'S1': s1,
            'R2': r2,
            'S2': s2
        })

    @staticmethod
    def fibonacci_retracement(prices):
        """
        Calcula os níveis de retração de Fibonacci.
        :param prices: Preços de fechamento.
        :return: Um DataFrame com os níveis de Fibonacci.
        """
        max_price = prices.max()
        min_price = prices.min()
        diff = max_price - min_price
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        retracements = [(max_price - diff * level) for level in levels]
        return pd.DataFrame({'Retracement Levels': retracements}, index=[f'{level * 100}%' for level in levels])