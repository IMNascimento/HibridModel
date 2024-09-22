# Hibridização de LSTM e Regressão Linear para Previsão do Mercado Financeiro

Este repositório contém o código-fonte dos experimentos desenvolvidos para o artigo que propõe a hibridização de Redes Neurais Long Short-Term Memory (LSTM) e Regressão Linear na previsão de preços do Bitcoin. Os experimentos foram realizados utilizando dados de séries temporais com uma abordagem de janela deslizante. A principal motivação deste estudo é comparar a eficácia de modelos tradicionais e de aprendizado profundo, além de investigar os ganhos com a combinação (hibridização) das duas abordagens.

## Estrutura do Repositório

- `main.py`: Script principal para executar os experimentos que utilizam LSTM, Regressão Linear e o modelo híbrido.
- `src/hyperParams.json`: Arquivo contendo os hiperparâmetros do experimento, como o tamanho da janela, número de unidades da LSTM, e taxa de aprendizado.
- `requirements.txt`: Arquivo listando todas as dependências necessárias para executar os scripts.
- `data/`: Pasta onde estão todas as base de dados.
- `src/data_processing.py`: Classe que separa e normaliza dados.
- `src/metrics_evalueation.py`: Classe para metricas de avaliação dos modelos.
- `src/model_training.py`: Classe para treino, salvamento dos modelos e execução.
- `src/plotting.py`: Classe para cuidar dos graficos gerados por cada modelo.
- `src/time_series_analyzer.py`: Classe que analiza e gera indicadores como as autocorrelações.
- `src/utils/csv_handler.py`: Classe para manipulação e carregamento de arquivos .csv.
- `src/utils/file_manager.py`: Classe para manipulação de arquivos e pastas para carregamento dos modelos salvos.
- `src/utils/logger.py`: Classe geradora de log de aplicação.
- `src/utils/technical_indicators.py`: Classe para geração e claculos de indicadores financeiros.

## Pré-requisitos

Para reproduzir os experimentos, você precisará ter o Python 3.11+ instalado em seu ambiente. Recomenda-se utilizar um ambiente virtual para gerenciar as dependências.

## Instalação

1. Clone este repositório em sua máquina local:

    ```bash
    git clone https://github.com/IMNascimento/HibridModel.git
    cd HibridModel
    ```

2. Crie um ambiente virtual e ative-o:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/MacOS
    venv\Scripts\activate  # Para Windows
    ```

3. Instale as dependências listadas no `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Executando os Experimentos

### Execução do Projeto
Para executar o experimento, basta rodar o script principal:

```bash
python main.py
```

### Alterando Hiperparâmetros
Os hiperparâmetros utilizados nos experimentos (como tamanho da janela, taxa de aprendizado, e número de épocas) estão configurados no arquivo hiperParams.json. Para ajustá-los, basta editar este arquivo.

Por exemplo, para alterar o tamanho da janela ou o número de unidades da LSTM, edite os valores no hyperParams.json:



```json
{
    "batch_size": 16,
    "units": 70,
    "dropout": 0.4,
    "epochs": 50,
    "input_window_size": 350,
    "output_window_size": 1,
    "datasets": [
        "data/bitcoin/hourly/BTCUSDT_hourly_2021_2021.csv",
        "data/bitcoin/hourly/BTCUSDT_hourly_2022_2022.csv",
        "data/bitcoin/hourly/BTCUSDT_hourly_2023_2023.csv"
    ]
}
```
Depois de ajustar os parâmetros, escolha a opção 10 do menu não precisando fazer a interrupção do programa e o sistema automaticamente ira  novamente para aplicar as mudanças.

## Resultados

Os resultados dos experimentos, incluindo gráficos de previsões e métricas como MSE, MAE, e R², serão salvos na pasta principal do projeto.


## Contribuições

Contribuições são bem-vindas! Se você tiver sugestões de melhorias ou encontrar problemas, fique à vontade para abrir uma issue ou enviar um pull request.

##  Citação

Se você utilizar este código ou artigo em sua pesquisa ou projeto, por favor, cite-o da seguinte forma:

**Formato ABNT:**
NASCIMENTO, Igor M. *Modelo híbrido LSTM e Regressão Linear para Previsão do Mercado Financeiro*. Disponível em:[imnascimento.github.io](https://imnascimento.github.io/Portifolio/assets/pdf/artigos/Modelo_h%C3%ADbrido_LSTM_e_Regress%C3%A3o_Linear_para_Previs%C3%A3o_do_Mercado_Financeiro.pdf). Acesso em: 21/09/2024.

**BibTeX:**
```bibtex
@misc{HibridModel,
  author = {Nascimento, Igor M.},
  title = {Modelo híbrido LSTM e Regressão Linear para Previsão do Mercado Financeiro},
  year = {2024},
  howpublished = {\url{https://imnascimento.github.io/Portifolio/assets/pdf/artigos/Modelo_h%C3%ADbrido_LSTM_e_Regress%C3%A3o_Linear_para_Previs%C3%A3o_do_Mercado_Financeiro.pdf}},
  note = {Acesso em: 21/09/2024}
}
```


## Contato

Para mais informações ou para acesso ao artigo, entre em contato através de [igor.muniz@estudante.ufjf.br](mailto:igor.muniz@estudante.ufjf.br).