import os

class FileManager:
    """
    Classe responsável por gerenciar a listagem de arquivos e pastas, além de navegar por diretórios.
    """
    
    @staticmethod
    def list_files(pasta):
        """
        Lista todos os arquivos disponíveis na pasta especificada e permite ao usuário selecionar um.
        
        :param pasta: Caminho da pasta onde os arquivos estão salvos.
        :return: Caminho completo do arquivo selecionado.
        """
        if not os.path.exists(pasta):
            print(f"A pasta '{pasta}' não foi encontrada.")
            return None
        
        arquivos_disponiveis = [f for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))]
        
        if len(arquivos_disponiveis) == 0:
            print("Nenhum arquivo foi encontrado na pasta.")
            return None

        print("\nArquivos disponíveis:")
        for idx, arquivo in enumerate(arquivos_disponiveis):
            print(f"{idx + 1} - {arquivo}")
        
        escolha = int(input("Digite o número do arquivo que deseja selecionar: ")) - 1

        if escolha < 0 or escolha >= len(arquivos_disponiveis):
            print("Escolha inválida. Tente novamente.")
            return None

        arquivo_escolhido = arquivos_disponiveis[escolha]
        caminho_arquivo = os.path.join(pasta, arquivo_escolhido)
        
        return caminho_arquivo

   