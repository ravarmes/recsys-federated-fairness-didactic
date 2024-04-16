import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance

class RecommendationNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, hidden_size):
        super(RecommendationNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(2 * embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        x = torch.cat((user_embedded, item_embedded), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Usando sigmoid para garantir valores entre 0 e 1
        # Transformando os valores para o intervalo desejado (1 a 5)
        x = 4 * x + 1
        return x.view(-1)

def carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo):
    df = pd.read_excel(caminho_do_arquivo) # Carregar os dados do arquivo Excel para um DataFrame do pandas
    df_com_zero = df.fillna(0) # Substituir valores NaN por zero
    df_dados = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
    tensor_dados = torch.tensor(df_dados.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
    return tensor_dados, df_dados.reset_index(drop=True)

def carregar_avaliacoes_do_arquivo_txt(caminho_do_arquivo):
    dados = np.loadtxt(caminho_do_arquivo, delimiter=',', dtype=np.float32)
    return torch.tensor(dados), dados
    
def treinar_modelo_global(modelo, avaliacoes, criterion, epochs=3, learning_rate=0.001):
    # optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)
    optimizer = optim.Adam(modelo.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    num_usuarios, num_itens = avaliacoes.shape
    usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    usuarios_ids, itens_ids = usuarios_ids.flatten(), itens_ids.flatten()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = modelo(usuarios_ids.long(), itens_ids.long()).view(num_usuarios, num_itens)
        loss = criterion(predictions, avaliacoes.float())
        loss.backward()
        optimizer.step()

def treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion, epochs=3, learning_rate=0.001):
    # Inicialização de dados e listas
    avaliacoes_final = avaliacoes_inicial.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes_inicial.size(0))] # criando uma cópia de modelo global inicial para cada usuário
    modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr = [], [], []
    
    NUMBER_ADVANTAGED_GROUP = 15
    NR_ADVANTAGED_GROUP, NR_DISADVANTAGED_GROUP = 20, 2

    num_usuarios, num_itens = avaliacoes_inicial.shape
    usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    usuarios_ids, itens_ids = usuarios_ids.flatten().long(), itens_ids.flatten().long()

    for i, modelo_cliente in enumerate(modelos_clientes):
        # print(f"=== Treinamento no Cliente {i + 1} ===")
        indices_nao_avaliados = (avaliacoes_inicial[i] == 0).nonzero(as_tuple=False).squeeze()

        indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP]]
        novas_avaliacoes = torch.randint(1, 6, (NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP,)).float()
        modelos_clientes_nr.append((i, NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP))

        avaliacoes_final[i, indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final_cliente = avaliacoes_inicial.clone()
        avaliacoes_final_cliente[i, indices_novas_avaliacoes] = novas_avaliacoes

        # optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate)
        optimizer_cliente = optim.Adam(modelo_cliente.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        for epoch in range(epochs):
            optimizer_cliente.zero_grad()
            predictions = modelo_cliente(usuarios_ids, itens_ids).view(num_usuarios, num_itens).float()
            loss_cliente = criterion(predictions, avaliacoes_final_cliente)
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(usuarios_ids, itens_ids).view(num_usuarios, num_itens)

        # Calculando as perdas individuais (lis) dos clientes em cada modelo de cliente
        avaliacoes_final_cliente_np = avaliacoes_final_cliente.numpy()
        avaliacoes_final_cliente_df = pd.DataFrame(avaliacoes_final_cliente_np)
        recomendacoes_cliente_np = recomendacoes_cliente.numpy()
        recomendacoes_cliente_df = pd.DataFrame(recomendacoes_cliente_np)
        omega_avaliacoes_final_cliente_df = (avaliacoes_final_cliente_df != 0)

        ilv_cliente = IndividualLossVariance(avaliacoes_final_cliente_df, omega_avaliacoes_final_cliente_df, 1)
        lis_cliente = ilv_cliente.get_losses(recomendacoes_cliente_df)

        modelos_clientes_rindv.append((i, lis_cliente[i])) # injustiças individuais do cliente local em seu respectivo modelo local
        modelos_clientes_loss.append((i, loss_cliente.item())) # perdas dos modelos locais

    return avaliacoes_final, modelos_clientes, modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr

def agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global, modelos_clientes):
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
            param_global.copy_(cliente_params.mean(dim=0))

def agregar_modelos_locais_ao_global_media_aritmetica_gradientes(modelo_global, modelos_clientes, learning_rate=0.034):
    with torch.no_grad():
        global_params = list(modelo_global.parameters())
        
        # Inicializar uma lista para armazenar a média dos gradientes para cada parâmetro
        gradientes_medios = [torch.zeros_like(param) for param in global_params]
        
        # Calcular a média dos gradientes para cada parâmetro
        for modelo_cliente in modelos_clientes:
            for i, param_cliente in enumerate(modelo_cliente.parameters()):
                if param_cliente.grad is not None:
                    gradientes_medios[i] += param_cliente.grad / len(modelos_clientes)
        
        # Atualizar os parâmetros do modelo global usando a média dos gradientes
        for i, param_global in enumerate(global_params):
            param_global -= learning_rate * gradientes_medios[i]


    num_clientes = len(modelos_clientes)
    peso_clientes = [0.05] * min(num_clientes, 15)  # Primeiros 15 modelos contribuem com 5%
    peso_restante = 0.95  # Peso para os modelos restantes

    if num_clientes > 15:
        peso_restante /= num_clientes - 15  # Distribui o restante igualmente entre os modelos restantes
        peso_clientes.extend([peso_restante] * (num_clientes - 15))

    with torch.no_grad():
        global_params = list(modelo_global.parameters())
        
        # Inicializar uma lista para armazenar a média ponderada dos gradientes para cada parâmetro
        gradientes_medios = [torch.zeros_like(param) for param in global_params]
        
        # Calcular a média ponderada dos gradientes para cada parâmetro
        for peso, modelo_cliente in zip(peso_clientes, modelos_clientes):
            for i, param_cliente in enumerate(modelo_cliente.parameters()):
                if param_cliente.grad is not None:
                    gradientes_medios[i] += peso * param_cliente.grad
        
        # Atualizar os parâmetros do modelo global usando a média ponderada dos gradientes
        for i, param_global in enumerate(global_params):
            param_global -= learning_rate * gradientes_medios[i]

def agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global, modelos_clientes, modelos_clientes_rindv):
    # Calcular o total de perdas
    total_perdas = sum(perda for _, perda in modelos_clientes_rindv)
    # print("total_perdas")
    # print(total_perdas)

    # Calcular os pesos de agregação baseados nas perdas
    pesos = [perda / total_perdas for _, perda in modelos_clientes_rindv]

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global, modelos_clientes, modelos_clientes_loss):
    # Calcular o total de perdas
    total_perdas = sum(perda for _, perda in modelos_clientes_loss)

    # Calcular os pesos de agregação baseados nas perdas
    pesos = [perda / total_perdas for _, perda in modelos_clientes_loss]
    # print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_loss")
    # print("modelos_clientes_loss")
    # print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global, modelos_clientes, modelos_clientes_nr):
    # Calcular o total_nr
    total_nr = sum(nr for _, nr in modelos_clientes_nr)
    # print("total_nr")
    # print(total_nr)

    # Calcular os pesos de agregação baseados nos valores inversos de total_nr
    pesos = [total_nr / nr if nr != 0 else 0 for _, nr in modelos_clientes_nr]

    # Normalizar os pesos para que a soma seja 1
    total_pesos = sum(pesos)
    pesos = [peso / total_pesos for peso in pesos]
    # print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_nr")
    # print("modelos_clientes_nr")
    # print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def main():
    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO INICIAL) ===")
    caminho_do_arquivo = 'X_MovieLens-1M.xlsx'
    avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)

    with torch.no_grad():
        # Preparar os índices de usuários e itens
        num_usuarios, num_itens = avaliacoes_inicial_tensor.shape
        usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
        usuarios_ids = usuarios_ids.reshape(-1)
        itens_ids = itens_ids.reshape(-1)

        # Assegurando que os índices sejam inteiros longos para compatibilidade com embedding layers
        usuarios_ids_long = usuarios_ids.long()
        itens_ids_long = itens_ids.long()

    # Exemplo de uso
    # num_usuarios = len(usuarios_ids_long)
    # num_itens = len(itens_ids_long)
    embedding_size = 64
    hidden_size = 128

    modelo_global_federado1 = RecommendationNN(num_usuarios, num_itens, embedding_size, hidden_size)
    criterion = nn.MSELoss() 

    treinar_modelo_global(modelo_global_federado1, avaliacoes_inicial_tensor, criterion)
    modelo_global_nao_federado = copy.deepcopy(modelo_global_federado1)
    modelo_global_federado2 = copy.deepcopy(modelo_global_federado1)
    modelo_global_federado3 = copy.deepcopy(modelo_global_federado1)
    modelo_global_federado4 = copy.deepcopy(modelo_global_federado1)

    with torch.no_grad():
        recomendacoes_inicial_tensor1 = modelo_global_federado1(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_inicial_tensor2 = modelo_global_federado2(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_inicial_tensor3 = modelo_global_federado3(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_inicial_tensor4 = modelo_global_federado4(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)

    
    for round in range(5):
        print(f"\n=== ROUND {round} ===")

        print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
        print("treinar_modelos_locais :: modelo_global_federado1")
        avaliacoes_final_tensor1, modelos_clientes1, modelos_clientes_rindv1, modelos_clientes_loss1, modelos_clientes_nr1 = treinar_modelos_locais(modelo_global_federado1, avaliacoes_inicial_tensor, criterion)
        print("treinar_modelos_locais :: modelo_global_federado2")
        avaliacoes_final_tensor2, modelos_clientes2, modelos_clientes_rindv2, modelos_clientes_loss2, modelos_clientes_nr2 = treinar_modelos_locais(modelo_global_federado2, avaliacoes_inicial_tensor, criterion)
        print("treinar_modelos_locais :: modelo_global_federado3")
        avaliacoes_final_tensor3, modelos_clientes3, modelos_clientes_rindv3, modelos_clientes_loss3, modelos_clientes_nr3 = treinar_modelos_locais(modelo_global_federado3, avaliacoes_inicial_tensor, criterion)
        print("treinar_modelos_locais :: modelo_global_federado4")
        avaliacoes_final_tensor4, modelos_clientes4, modelos_clientes_rindv4, modelos_clientes_loss4, modelos_clientes_nr4 = treinar_modelos_locais(modelo_global_federado4, avaliacoes_inicial_tensor, criterion)

        # print("\nmodelos_clientes_rindv")
        # print(modelos_clientes_rindv)
        # print("\nmodelos_clientes_loss")
        # print(modelos_clientes_loss)
        # print("\nmodelos_clientes_nr")
        # print(modelos_clientes_nr)

        print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL - AGREGAÇÃO) ===")
        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos :: modelo_global_federado1")
        agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado1, modelos_clientes1)
        # agregar_modelos_locais_ao_global_media_aritmetica_gradientes(modelo_global_federado1, modelos_clientes)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_rindv :: modelo_global_federado2")
        agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global_federado2, modelos_clientes2, modelos_clientes_rindv2)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_loss :: modelo_global_federado3")
        agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global_federado3, modelos_clientes3, modelos_clientes_loss3)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_nr :: modelo_global_federado4")
        agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global_federado4, modelos_clientes4, modelos_clientes_nr4)

        # treinar_modelo_global(modelo_global_federado1, avaliacoes_final_tensor1, criterion) 
        # treinar_modelo_global(modelo_global_federado2, avaliacoes_final_tensor2, criterion) 
        # treinar_modelo_global(modelo_global_federado3, avaliacoes_final_tensor3, criterion) 
        # treinar_modelo_global(modelo_global_federado4, avaliacoes_final_tensor4, criterion) 
        treinar_modelo_global(modelo_global_nao_federado, avaliacoes_final_tensor1, criterion) 

    with torch.no_grad():
        recomendacoes_final_tensor1 = modelo_global_federado1(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_tensor2 = modelo_global_federado2(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_tensor3 = modelo_global_federado3(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_tensor4 = modelo_global_federado4(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_modelo_global_nao_federado_tensor = modelo_global_nao_federado(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)


    
    print("\n=== MEDIDA DE JUSTIÇA ===")
    avaliacoes_inicial_np = avaliacoes_inicial_tensor.numpy()
    avaliacoes_inicial_df = pd.DataFrame(avaliacoes_inicial_np)
    avaliacoes_final_np = avaliacoes_final_tensor1.numpy()
    avaliacoes_final_df = pd.DataFrame(avaliacoes_final_np)
    recomendacoes_inicial_np = recomendacoes_inicial_tensor1.numpy()
    recomendacoes_inicial_df = pd.DataFrame(recomendacoes_inicial_np)
    recomendacoes_final_np1 = recomendacoes_final_tensor1.numpy()
    recomendacoes_final_df1 = pd.DataFrame(recomendacoes_final_np1)
    recomendacoes_final_np2 = recomendacoes_final_tensor2.numpy()
    recomendacoes_final_df2 = pd.DataFrame(recomendacoes_final_np2)
    recomendacoes_final_np3 = recomendacoes_final_tensor3.numpy()
    recomendacoes_final_df3 = pd.DataFrame(recomendacoes_final_np3)
    recomendacoes_final_np4 = recomendacoes_final_tensor4.numpy()
    recomendacoes_final_df4 = pd.DataFrame(recomendacoes_final_np4)
    recomendacoes_modelo_global_nao_federado_np = recomendacoes_modelo_global_nao_federado_tensor.numpy()
    recomendacoes_modelo_global_nao_federado_df = pd.DataFrame(recomendacoes_modelo_global_nao_federado_np)

    omega_inicial = (avaliacoes_inicial_df != 0)
    omega_final = (avaliacoes_final_df != 0)    

    # To capture polarization, we seek to measure the extent to which the user ratings disagree
    polarization = Polarization()
    Rpol_inicial = polarization.evaluate(recomendacoes_inicial_df)
    Rpol_final1 = polarization.evaluate(recomendacoes_final_df1)
    Rpol_final2 = polarization.evaluate(recomendacoes_final_df2)
    Rpol_final3 = polarization.evaluate(recomendacoes_final_df3)
    Rpol_final4 = polarization.evaluate(recomendacoes_final_df4)
    Rpol_final_nao_federado = polarization.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f"\nPolarization Inicial (Rpol)                : {Rpol_inicial:.9f}")
    print(f"Polarization Final   (Rpol [1])            : {Rpol_final1:.9f}")
    print(f"Polarization Final   (Rpol [2])            : {Rpol_final2:.9f}")
    print(f"Polarization Final   (Rpol [3])            : {Rpol_final3:.9f}")
    print(f"Polarization Final   (Rpol [4])            : {Rpol_final4:.9f}")
    print(f"Polarization Final   (Rpol [Não Federado]) : {Rpol_final_nao_federado:.9f}")

    ilv_inicial = IndividualLossVariance(avaliacoes_inicial_df, omega_inicial, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final1 = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final2 = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final3 = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final4 = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_nao_federado = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    Rindv_inicial = ilv_inicial.evaluate(recomendacoes_inicial_df)
    Rindv_final1 = ilv_final1.evaluate(recomendacoes_final_df1)
    Rindv_final2 = ilv_final2.evaluate(recomendacoes_final_df2)
    Rindv_final3 = ilv_final3.evaluate(recomendacoes_final_df3)
    Rindv_final4 = ilv_final4.evaluate(recomendacoes_final_df4)
    Rindv_final_nao_federado = ilv_final_nao_federado.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f"\nIndividual Loss Variance (Rindv Inicial)                            : {Rindv_inicial:.9f}")
    print(f"Individual Loss Variance (Rindv Final [1 :: Média Aritmética     ]) : {Rindv_final1:.9f}")
    print(f"Individual Loss Variance (Rindv Final [2 :: Média Ponderada Rindv]) : {Rindv_final2:.9f}")
    print(f"Individual Loss Variance (Rindv Final [3 :: Média Ponderada Loss ]) : {Rindv_final3:.9f}")
    print(f"Individual Loss Variance (Rindv Final [4 :: Média Ponderada NR   ]) : {Rindv_final4:.9f}")
    print(f"Individual Loss Variance (Rindv Final [Não Federado              ]) : {Rindv_final_nao_federado:.9f}")

    # # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # # advantaged group: 5% users with the highest number of item ratings
    # # disadvantaged group: 95% users with the lowest number of item ratings
    modelos_clientes_rindv_ordenados = sorted(modelos_clientes_rindv1, key=lambda x: x[1], reverse=False)
    list_users_rindv = [i for i, _ in modelos_clientes_rindv_ordenados]
    advantaged_group_rindv = list_users_rindv[0:15]
    disadvantaged_group_rindv = list_users_rindv[15:300]
    G_RINDV = {1: advantaged_group_rindv, 2: disadvantaged_group_rindv}

    modelos_clientes_loss_ordenados = sorted(modelos_clientes_loss3, key=lambda x: x[1], reverse=False)
    list_users_loss = [i for i, _ in modelos_clientes_loss_ordenados]
    advantaged_group_loss = list_users_loss[0:15]
    disadvantaged_group_loss = list_users_loss[15:300]
    G_LOSS = {1: advantaged_group_loss, 2: disadvantaged_group_loss}
    
    modelos_clientes_nr_ordenados = sorted(modelos_clientes_nr4, key=lambda x: x[1], reverse=True)
    list_users_nr = [i for i, _ in modelos_clientes_nr_ordenados]
    advantaged_group_nr = list_users_nr[0:15]
    disadvantaged_group_nr = list_users_nr[15:300]
    G_NR = {1: advantaged_group_nr, 2: disadvantaged_group_nr}

    print("G_RINDV")
    print(G_RINDV[1])

    print("G_LOSS")
    print(G_LOSS[1])

    print("G_NR")
    print(G_NR[1])

    glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final1_rindv = GroupLossVariance(avaliacoes_final_df, omega_final, G_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final1_loss = GroupLossVariance(avaliacoes_final_df, omega_final, G_LOSS, 1) #axis = 1 (0 rows e 1 columns)
    glv_final1_nr = GroupLossVariance(avaliacoes_final_df, omega_final, G_NR, 1) #axis = 1 (0 rows e 1 columns)
    glv_final2 = GroupLossVariance(avaliacoes_final_df, omega_final, G_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final3 = GroupLossVariance(avaliacoes_final_df, omega_final, G_LOSS, 1) #axis = 1 (0 rows e 1 columns)
    glv_final4 = GroupLossVariance(avaliacoes_final_df, omega_final, G_NR, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_nao_federado1_rindv = GroupLossVariance(avaliacoes_final_df, omega_final, G_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_nao_federado2_loss = GroupLossVariance(avaliacoes_final_df, omega_final, G_LOSS, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_nao_federado3_nr = GroupLossVariance(avaliacoes_final_df, omega_final, G_NR, 1) #axis = 1 (0 rows e 1 columns)
    RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_df)
    RgrpNR_final1_rindv = glv_final1_rindv.evaluate(recomendacoes_final_df1)
    RgrpNR_final1_loss = glv_final1_loss.evaluate(recomendacoes_final_df1)
    RgrpNR_final1_nr = glv_final1_nr.evaluate(recomendacoes_final_df1)
    RgrpNR_final2 = glv_final2.evaluate(recomendacoes_final_df2)
    RgrpNR_final3 = glv_final3.evaluate(recomendacoes_final_df3)
    RgrpNR_final4 = glv_final4.evaluate(recomendacoes_final_df4)
    RgrpNR_final_nao_federado1_rindv = glv_final_nao_federado1_rindv.evaluate(recomendacoes_modelo_global_nao_federado_df)
    RgrpNR_final_nao_federado2_loss = glv_final_nao_federado2_loss.evaluate(recomendacoes_modelo_global_nao_federado_df)
    RgrpNR_final_nao_federado3_nr = glv_final_nao_federado3_nr.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f"\nGroup Loss Variance (Rgrp Inicial)                              : {RgrpNR_inicial:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética Rindv ]) : {RgrpNR_final1_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética Loss  ]) : {RgrpNR_final1_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética NR    ]) : {RgrpNR_final1_nr:.9f}")
    print(f"Group Loss Variance (Rgrp Final [2 :: Média Ponderada Rindv  ]) : {RgrpNR_final2:.9f}")
    print(f"Group Loss Variance (Rgrp Final [3 :: Média Ponderada Loss   ]) : {RgrpNR_final3:.9f}")
    print(f"Group Loss Variance (Rgrp Final [4 :: Média Ponderada NR     ]) : {RgrpNR_final4:.9f}")
    print(f"Group Loss Variance (Rgrp Final [Não Federado :: Rindv       ]) : {RgrpNR_final_nao_federado1_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [Não Federado :: Loss        ]) : {RgrpNR_final_nao_federado2_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [Não Federado :: NR          ]) : {RgrpNR_final_nao_federado3_nr:.9f}")

    rmse_inicial = RMSE(avaliacoes_inicial_df, omega_inicial)
    result_inicial = rmse_inicial.evaluate(recomendacoes_inicial_df)
    rmse_final1 = RMSE(avaliacoes_final_df, omega_final)
    result_final1 = rmse_final1.evaluate(recomendacoes_final_df1)
    rmse_final2 = RMSE(avaliacoes_final_df, omega_final)
    result_final2 = rmse_final2.evaluate(recomendacoes_final_df2)
    rmse_final3 = RMSE(avaliacoes_final_df, omega_final)
    result_final3 = rmse_final3.evaluate(recomendacoes_final_df3)
    rmse_final4 = RMSE(avaliacoes_final_df, omega_final)
    result_final4 = rmse_final4.evaluate(recomendacoes_final_df4)
    rmse_final_nao_federado = RMSE(avaliacoes_final_df, omega_final)
    result_final_nao_federado = rmse_final_nao_federado.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f'\nRMSE Inicial                            : {result_inicial:.9f}')
    print(f'RMSE Final [1 :: Média Aritmética     ] : {result_final1:.9f}')
    print(f'RMSE Final [2 :: Média Ponderada Rindv] : {result_final2:.9f}')
    print(f'RMSE Final [3 :: Média Ponderada Loss ] : {result_final3:.9f}')
    print(f'RMSE Final [4 :: Média Ponderada NR   ] : {result_final4:.9f}')
    print(f'RMSE Final [Não Federado              ] : {result_final_nao_federado:.9f}')

    # avaliacoes_inicial_df.to_excel("avaliacoes_inicial.xlsx", index=False)
    # avaliacoes_final_df.to_excel("avaliacoes_final.xlsx", index=False)
    # recomendacoes_inicial_df.to_excel("recomendacoes_inicial.xlsx", index=False)
    # recomendacoes_final_df1.to_excel("recomendacoes_final_df1.xlsx", index=False)
    # recomendacoes_final_df2.to_excel("recomendacoes_final_df2.xlsx", index=False)
    # recomendacoes_final_df3.to_excel("recomendacoes_final_df3.xlsx", index=False)
    # recomendacoes_final_df4.to_excel("recomendacoes_final_df4.xlsx", index=False)
    # recomendacoes_modelo_global_nao_federado_df.to_excel("recomendacoes_modelo_global_nao_federado_df.xlsx", index=False)

if __name__ == "__main__":
    main()
