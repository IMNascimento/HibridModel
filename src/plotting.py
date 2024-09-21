import matplotlib.pyplot as plt
from src.utils.logger import Logger

class ResultPlotter:
    
    @staticmethod
    def plot_comparison(y_real, y_pred, title='Comparison', save_path=None):
        """
        Plota a comparação entre os valores reais e os preditos.
        
        :param y_real: Valores reais.
        :param y_pred: Valores preditos.
        :param title: Título do gráfico.
        :param save_path: Caminho para salvar o gráfico.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(y_real, label='Real', color='blue')
        plt.plot(y_pred, label='Predicted', color='red')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            Logger.info(f"Gráfico salvo em: {save_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_autocorrelation(autocorrelations, save_path=None):
        """
        Plota e salva o gráfico de autocorrelação.
        
        :param autocorrelations: Dicionário com os tamanhos de janela e autocorrelações.
        :param save_path: Caminho para salvar o gráfico.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(list(autocorrelations.keys()), list(autocorrelations.values()), marker='o')
        plt.title('Autocorrelação para Diferentes Tamanhos de Janela')
        plt.xlabel('Tamanho da Janela')
        plt.ylabel('Autocorrelação')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            Logger.info(f"Gráfico salvo em: {save_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_correlation(correlations, title, save_path=None):
        """
        Plota e salva a correlação entre indicadores técnicos e o preço de fechamento.
        
        :param correlations: Séries de correlações.
        :param title: Título do gráfico.
        :param save_path: Caminho para salvar o gráfico.
        """
        plt.figure(figsize=(10, 6))
        correlations.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.xlabel('Indicadores Técnicos')
        plt.ylabel('Correlação')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            Logger.info(f"Gráfico salvo em: {save_path}")
        else:
            plt.show()

