import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DNNRecomendacao(nn.Module):
    def __init__(self, num_usuarios, num_itens, embedding_dim=64, hidden_dim=128):
        super(DNNRecomendacao, self).__init__()
        self.usuario_embedding = nn.Embedding(num_usuarios, embedding_dim)
        self.item_embedding = nn.Embedding(num_itens, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, usuario, item):
        usuario_embedded = self.usuario_embedding(usuario)
        item_embedded = self.item_embedding(item)
        x = torch.cat([usuario_embedded, item_embedded], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ClienteSistemaRecomendacao:
    def __init__(self, modelo_global, dados_locais):
        self.modelo = modelo_global
        self.dados_locais = dados_locais
        
    def gerar_novas_avaliacoes(self, quantidade):
        itens_avaliados = set([dado[1] for dado in self.dados_locais])
        itens_disponiveis = set(range(len(self.modelo[0]))).difference(itens_avaliados)
        
        novas_avaliacoes = []
        for _ in range(quantidade):
            if itens_disponiveis:  # Verifica se há itens disponíveis para escolher
                item_novo = np.random.choice(list(itens_disponiveis))
                usuario = np.random.randint(len(self.modelo))
                avaliacao = np.random.randint(1, 6)
                novas_avaliacoes.append((usuario, item_novo, avaliacao))
        
        self.dados_locais.extend(novas_avaliacoes)

    
    def treinar_localmente(self):
        modelo_local = DNNRecomendacao(num_usuarios=self.modelo.shape[0], num_itens=self.modelo.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(modelo_local.parameters(), lr=0.01)
        
        for _ in range(20):  # 20 épocas de treinamento
            for novo_dado in self.dados_locais:
                if len(novo_dado) != 3:  # Verifica se tem 3 elementos
                    continue  # Pula para o próximo dado se não tiver 3 elementos
                
                usuario_id, item_id, avaliacao = novo_dado
                
                output = modelo_local(torch.tensor([usuario_id]), torch.tensor([item_id]))
                loss = criterion(output, torch.tensor([[avaliacao]], dtype=torch.float))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return modelo_local

class ServidorSistemaRecomendacao:
    def __init__(self):
        self.modelo_global = None
    
    def iniciar_modelo(self, arquivo_excel):
        dados_excel = pd.read_excel(arquivo_excel, header=None).fillna(0)
        
        if dados_excel.shape != (300, 1000):
            raise ValueError("A matriz deve ter dimensões 300x1000")
        
        self.modelo_global = dados_excel.values.astype(np.float32)

    def agregar_modelos_locais(self, modelos_locais):
        num_modelos_locais = len(modelos_locais)
        novo_modelo_global = torch.zeros_like(torch.tensor(self.modelo_global))
        
        for modelo_local in modelos_locais:
            for param_global, param_local in zip(self.modelo_global, modelo_local.parameters()):
                param_global.add_(param_local)
        
        self.modelo_global = novo_modelo_global / num_modelos_locais

# Carregar arquivo Excel e iniciar o servidor
servidor = ServidorSistemaRecomendacao()
servidor.iniciar_modelo('X.xlsx')

# Criar clientes com avaliações já realizadas
print("carregando avaliações de clientes")
clientes = []
for i in range(300):
    print(f"cliente {i}")
    avaliacoes_indices = np.nonzero(servidor.modelo_global) # Encontrar índices não nulos
    indices = list(zip(avaliacoes_indices[0], avaliacoes_indices[1])) # Lista de índices
    avaliacoes_cliente = [(usuario, item, servidor.modelo_global[usuario][item]) for usuario, item in indices]
    clientes.append(ClienteSistemaRecomendacao(servidor.modelo_global, avaliacoes_cliente))

# Treinamento e agregação de modelos locais
for round in range(2):
    print(f"round: {round}")
    modelos_locais = []
    i = 0
    for cliente in clientes:
        i = i + 1
        cliente.modelo = servidor.modelo_global  # Compartilhar o modelo global atualizado
        cliente.gerar_novas_avaliacoes(1)
        modelo_local_atualizado = cliente.treinar_localmente()
        print(f"Treinando cliente: {i}")
        modelos_locais.append(modelo_local_atualizado)
    
    print("Agregando modelos locais ao modelo global")
    servidor.agregar_modelos_locais(modelos_locais)

modelo_global_final = servidor.modelo_global
