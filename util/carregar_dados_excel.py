import pandas as pd
import torch

def carregar_dados_movie_lens(caminho_do_arquivo):
    # Carregar os dados do arquivo Excel para um DataFrame do pandas
    df = pd.read_excel(caminho_do_arquivo)
    
    # Substituir valores NaN por zero
    df_com_zero = df.fillna(0)
    
    # Selecionar apenas as colunas de dados com as avaliações dos filmes
    dados_filmes = df_com_zero.iloc[:, 1:]
    
    # Converter o DataFrame para um tensor PyTorch
    tensor_dados = torch.tensor(dados_filmes.values, dtype=torch.float32)
    
    return tensor_dados

# Exemplo de utilização do método para carregar os dados
caminho_arquivo = 'X_MovieLens-1M.xlsx'  # Substitua com o caminho correto do arquivo
tensor_dados_movie_lens = carregar_dados_movie_lens(caminho_arquivo)

# Exibindo o tensor dos dados carregados
print(tensor_dados_movie_lens)
