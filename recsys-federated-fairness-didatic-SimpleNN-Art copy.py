import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.activation(self.layer2(x)) * 4 + 1  # Scale sigmoid output to range [1, 5]
        return x

def calcular_diferencas_modelos(modelo_a, modelo_b):
    l1_diferenca, l2_diferenca = 0.0, 0.0
    for param_a, param_b in zip(modelo_a.parameters(), modelo_b.parameters()):
        # Calculando a diferença L1 (norma Manhattan / distância absoluta)
        l1_diferenca += torch.norm((param_a - param_b).abs(), p=1).item()

        # Calculando a diferença L2 (norma euclidiana)
        l2_diferenca += torch.norm(param_a - param_b, p=2).item()

    return l1_diferenca, l2_diferenca

def carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo):
    df = pd.read_excel(caminho_do_arquivo) # Carregar os dados do arquivo Excel para um DataFrame do pandas
    df_com_zero = df.fillna(0) # Substituir valores NaN por zero
    df_dados = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
    tensor_dados = torch.tensor(df_dados.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
    return tensor_dados, df_dados.reset_index(drop=True)

def carregar_avaliacoes_do_arquivo_txt(caminho_do_arquivo):
    dados = np.loadtxt(caminho_do_arquivo, delimiter=',', dtype=np.float32)
    return torch.tensor(dados), dados
    
def treinar_modelo_global(modelo, avaliacoes, criterion, epochs=1, learning_rate=0.034):
    """
    Treina o modelo global usando uma matriz de avaliações.
    
    Args:
        modelo (torch.nn.Module): O modelo de rede neural a ser treinado.
        avaliacoes (torch.Tensor): Um tensor contendo as avaliações dos usuários sobre os itens.
        criterion (torch.nn.modules.loss._Loss): A função de perda utilizada para o treinamento.
        epochs (int, optional): Número de épocas para o treinamento. Padrão é 50.
        learning_rate (float, optional): Taxa de aprendizado para o otimizador SGD. Padrão é 0.01.
        
    Descrição:
        Esta função treina o modelo global utilizando a matriz de avaliações fornecida.
        Utiliza o otimizador SGD (Descida do Gradiente Estocástica) com a taxa de aprendizado
        especificada. A função de perda calculada a cada época é baseada na diferença entre
        as saídas do modelo e as avaliações reais. Os parâmetros do modelo são atualizados
        em cada passo do treinamento para minimizar a função de perda.
    """
    optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(avaliacoes)
        loss = criterion(output, avaliacoes)
        loss.backward()
        optimizer.step()

def treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion, epochs=1, learning_rate=0.034):
    avaliacoes_final = avaliacoes_inicial.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes_inicial.size(0))]
    modelos_clientes_rindv = [] # injustiças individuais de cada cliente local em seu respectivo modelo local
    modelos_clientes_loss = [] # perdas dos modelos locais de cada cliente local
    modelos_clientes_nr = [] # número de avaliações de cada cliente local

    NR_ADVANTAGED_GROUP = 120      # número de avaliações geradas para os clientes do grupo dos favorecidos
    NR_DISADVANTAGED_GROUP = 10  # número de avaliações geradas para os clientes do grupo dos desfavorecidos

    print(f"\nAntes do treinamento param local cliente 0 parameters[0]\n")
    print(list(modelos_clientes[0].parameters())[0])

    print(f"\nAntes do treinamento param local cliente 1 parameters[0]\n")
    print(list(modelos_clientes[1].parameters())[0])

    print(f"\nAntes do treinamento param local cliente 200 parameters[0]\n")
    print(list(modelos_clientes[200].parameters())[0])

    for i, modelo_cliente in enumerate(modelos_clientes):
        # Gerar índices de itens não avaliados
        indices_nao_avaliados = (avaliacoes_inicial[i] == 0).nonzero().squeeze()
        
        if i < 15:
            # Selecionar NR_ADVANTAGED_GROUP índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_ADVANTAGED_GROUP]]
            # Gerar NR_ADVANTAGED_GROUP novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (NR_ADVANTAGED_GROUP,)).float()
            modelos_clientes_nr.append((i, NR_ADVANTAGED_GROUP))
        else:
            # Selecionar NR_DISADVANTAGED_GROUP índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_DISADVANTAGED_GROUP]]
            # Gerar NR_DISADVANTAGED_GROUP novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (NR_DISADVANTAGED_GROUP,)).float()
            modelos_clientes_nr.append((i, NR_DISADVANTAGED_GROUP))

        # Atualizar avaliações iniciais com novas avaliações
        avaliacoes_final[i][indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final_cliente = avaliacoes_inicial.clone()  # Usar clone para manter as avaliações iniciais
        avaliacoes_final_cliente[i][indices_novas_avaliacoes] = novas_avaliacoes

        if i == 0:
            print("\navaliacoes_final_cliente 0\n")
            print(avaliacoes_final_cliente)

        if i == 1:
            print("\navaliacoes_final_cliente 1\n")
            print(avaliacoes_final_cliente)

        if i == 200:
            print("\navaliacoes_final_cliente 200\n")
            print(avaliacoes_final_cliente)

        print(f"=== Treinamento no Cliente {i + 1} ===")

        # modelo_cliente_local = copy.deepcopy(modelo_cliente)  # Clonando o modelo_cliente para um novo objeto
        # optimizer_cliente = optim.SGD(modelo_cliente_local.parameters(), lr=learning_rate)
        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate)
        
        for _ in range(epochs):
            optimizer_cliente.zero_grad()
            output_cliente = modelo_cliente(avaliacoes_final_cliente)
            loss_cliente = criterion(output_cliente, avaliacoes_final_cliente)
            loss_cliente.backward()
            optimizer_cliente.step()

        if i == 0:
            print(f"\nDepois do treinamento param local cliente {i} parameters[0]\n")
            print(list(modelos_clientes[i].parameters())[0])

        if i == 1:
            print(f"\nDepois do treinamento param local cliente {i} parameters[0]\n")
            print(list(modelos_clientes[i].parameters())[0])

        if i == 200:
            print(f"\nDepois do treinamento param local cliente {i} parameters[0]\n")
            print(list(modelos_clientes[i].parameters())[0])

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(avaliacoes_final_cliente)

        if i == 0:
            print("\nrecomendacoes_cliente 0\n")
            print(recomendacoes_cliente)

        if i == 1:
            print("\nrecomendacoes_cliente 1\n")
            print(recomendacoes_cliente)

        if i == 200:
            print("\nrecomendacoes_cliente 200\n")
            print(recomendacoes_cliente)
        
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

    print("\n\nverificando os modelos de clientes 1 e 2\n")
    print(modelos_clientes[0] is modelos_clientes[1])

    # Retorna ambos: avaliações finais, avaliações finais por cliente, modelos dos clientes e perdas dos modelos (baseado na injustiça individual li)
    return avaliacoes_final, modelos_clientes, modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr

def agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global, modelos_clientes):
    """
    Atualiza os parâmetros do modelo global com a média dos parâmetros dos modelos locais.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.

    Descrição:
        Esta função percorre cada parâmetro (por exemplo, pesos e vieses) do modelo global e
        atualiza seus valores com a média dos valores dos parâmetros correspondentes dos
        modelos locais. Essa abordagem assume que uma média simples dos parâmetros pode
        levar a um modelo global mais genérico e robusto, incorporando o aprendizado obtido
        de várias fontes de dados locais.

        O processo de agregação é uma etapa fundamental no aprendizado federado, permitindo
        que o modelo global beneficie-se do aprendizado distribuído sem a necessidade de
        compartilhar diretamente os dados locais, preservando assim a privacidade dos dados.

        É importante que esta função seja chamada após o treinamento dos modelos locais e
        antes da próxima rodada de distribuição do modelo global atualizado para treinamento
        local adicional ou para inferência.
    """
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
            param_global.copy_(cliente_params.mean(dim=0))

def agregar_modelos_locais_ao_global_media_aritmetica_gradientes(modelo_global, modelos_clientes, learning_rate=0.034):
    """
    Atualiza os parâmetros do modelo global com a média dos gradientes dos modelos locais.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.

    Descrição:
        Esta função percorre cada parâmetro (por exemplo, pesos e vieses) do modelo global e
        atualiza seus valores com a média dos gradientes dos parâmetros correspondentes dos
        modelos locais. A ideia é que, em vez de atualizar o modelo global com a média direta
        dos parâmetros dos modelos locais, utilizamos os gradientes (derivadas da função de
        perda em relação aos parâmetros) para fazer uma atualização baseada em como cada
        modelo local "aprendeu" a partir de seus dados.

        Este método é uma abordagem alternativa ao processo padrão de agregação em
        aprendizado federado, permitindo um ajuste mais fino do modelo global com base nas
        tendências de aprendizado locais.

        É importante ressaltar que, para que esta função funcione como esperado, os modelos
        locais devem ter seus gradientes retidos (não zerados) após o treinamento.
    """
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


    """
    Atualiza os parâmetros do modelo global com a média ponderada dos gradientes dos modelos locais.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.
        learning_rate (float): Taxa de aprendizado para a atualização dos parâmetros do modelo global.

    Descrição:
        Esta função percorre cada parâmetro do modelo global e atualiza seus valores com a média ponderada dos gradientes
        dos parâmetros correspondentes dos modelos locais. A ponderação é determinada pela regra de justiça, onde os
        primeiros 15 modelos contribuem com 5% da agregação, enquanto os modelos restantes contribuem com 95%.

        Este método é uma abordagem alternativa ao processo padrão de agregação em aprendizado federado, permitindo um
        ajuste mais fino do modelo global com base nas tendências de aprendizado locais e considerando justiça na agregação.

        É importante ressaltar que, para que esta função funcione como esperado, os modelos locais devem ter seus gradientes
        retidos (não zerados) após o treinamento.
    """
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
    # print(f"\n\nTAMANHO")
    # total_params = sum(p.numel() for p in modelo_global.parameters())
    # print(f"Total de parâmetros no modelo: {len(enumerate(modelo_global.parameters()))}")
    print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_rindv\n")
    # # imprimindo os parâmetros de um modelo
    # print("\nIMPRIMINDO PARÂMETROS DO MODELO GLOBAL ANTES DA AGREGAÇÃO")
    # for parametro in modelo_global.parameters():
    #     print(parametro)

    # Calcular o total de perdas
    total_perdas = sum(perda for _, perda in modelos_clientes_rindv)
    # print("total_perdas")
    # print(total_perdas)

    # Calcular os pesos de agregação baseados nas perdas
    pesos = [perda / total_perdas for _, perda in modelos_clientes_rindv]
    # print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_rindv")
    # print("modelos_clientes_rindv")
    # print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            if i == 0 or i == 3:
                print(f"\nparam_global {i}\n")
                print(param_global)
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                if ((i == 0 or i == 3) and (j == 0 or j == 299)):
                    print(f"\nparam local cliente {j} param {i}\n")
                    print(list(modelos_clientes[j].parameters())[i])
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

    # # imprimindo os parâmetros de um modelo
    # print("\nIMPRIMINDO PARÂMETROS DO MODELO GLOBAL DEPOIS DA AGREGAÇÃO")
    # for parametro in modelo_global.parameters():
    #     print(parametro)

    # print("\nIMPRIMINDO PARÂMETROS DO MODELO LOCAL 1")
    # for parametro in modelos_clientes[0].parameters():
    #     print(parametro)

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

    numero_de_usuarios = avaliacoes_inicial_tensor.shape[0]
    numero_de_itens = avaliacoes_inicial_tensor.shape[1]

    # modelo_global_federado1 = SimpleNN(numero_de_itens, 2, numero_de_itens)
    modelo_global_federado1 = SimpleNN(numero_de_itens, 20, numero_de_itens)
    criterion = nn.MSELoss() 

    treinar_modelo_global(modelo_global_federado1, avaliacoes_inicial_tensor, criterion)
    modelo_global_nao_federado = copy.deepcopy(modelo_global_federado1)
    modelo_global_federado2 = copy.deepcopy(modelo_global_federado1)
    modelo_global_federado3 = copy.deepcopy(modelo_global_federado1)
    modelo_global_federado4 = copy.deepcopy(modelo_global_federado1)

    with torch.no_grad():
        recomendacoes_inicial_tensor = modelo_global_federado1(avaliacoes_inicial_tensor)

    print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
    avaliacoes_final_tensor, modelos_clientes, modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr = treinar_modelos_locais(modelo_global_federado1, avaliacoes_inicial_tensor, criterion)

    print("\nmodelos_clientes_rindv")
    print(modelos_clientes_rindv)
    print("\nmodelos_clientes_loss")
    print(modelos_clientes_loss)
    print("\nmodelos_clientes_nr")
    print(modelos_clientes_nr)

    l1_dif, l2_dif = calcular_diferencas_modelos(modelos_clientes[0], modelos_clientes[1])
    print(f"Diferença L1 total entre os modelos: {l1_dif}")
    print(f"Diferença L2 total entre os modelos: {l2_dif}")

    l1_dif, l2_dif = calcular_diferencas_modelos(modelos_clientes[0], modelos_clientes[200])
    print(f"Diferença L1 total entre os modelos: {l1_dif}")
    print(f"Diferença L2 total entre os modelos: {l2_dif}")

    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL - AGREGAÇÃO) ===")
    agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado1, modelos_clientes)
    # agregar_modelos_locais_ao_global_media_aritmetica_gradientes(modelo_global_federado1, modelos_clientes)
    agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global_federado2, modelos_clientes, modelos_clientes_rindv)
    agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global_federado3, modelos_clientes, modelos_clientes_loss)
    agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global_federado4, modelos_clientes, modelos_clientes_nr)

    treinar_modelo_global(modelo_global_nao_federado, avaliacoes_final_tensor, criterion) # Simulando um modelo não federado

    with torch.no_grad():
        recomendacoes_final_tensor1 = modelo_global_federado1(avaliacoes_final_tensor)
        recomendacoes_final_tensor2 = modelo_global_federado2(avaliacoes_final_tensor)
        recomendacoes_final_tensor3 = modelo_global_federado3(avaliacoes_final_tensor)
        recomendacoes_final_tensor4 = modelo_global_federado4(avaliacoes_final_tensor)
        recomendacoes_modelo_global_nao_federado_tensor = modelo_global_nao_federado(avaliacoes_final_tensor)

    
    print("\n=== MEDIDA DE JUSTIÇA ===")
    avaliacoes_inicial_np = avaliacoes_inicial_tensor.numpy()
    avaliacoes_inicial_df = pd.DataFrame(avaliacoes_inicial_np)
    avaliacoes_final_np = avaliacoes_final_tensor.numpy()
    avaliacoes_final_df = pd.DataFrame(avaliacoes_final_np)
    recomendacoes_inicial_np = recomendacoes_inicial_tensor.numpy()
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
    modelos_clientes_rindv_ordenados = sorted(modelos_clientes_rindv, key=lambda x: x[1], reverse=False)
    list_users_rindv = [i for i, _ in modelos_clientes_rindv_ordenados]
    advantaged_group_rindv = list_users_rindv[0:15]
    disadvantaged_group_rindv = list_users_rindv[15:300]
    G_RINDV = {1: advantaged_group_rindv, 2: disadvantaged_group_rindv}

    modelos_clientes_loss_ordenados = sorted(modelos_clientes_loss, key=lambda x: x[1], reverse=False)
    list_users_loss = [i for i, _ in modelos_clientes_loss_ordenados]
    advantaged_group_loss = list_users_loss[0:15]
    disadvantaged_group_loss = list_users_loss[15:300]
    G_LOSS = {1: advantaged_group_loss, 2: disadvantaged_group_loss}
    
    modelos_clientes_nr_ordenados = sorted(modelos_clientes_nr, key=lambda x: x[1], reverse=True)
    list_users_nr = [i for i, _ in modelos_clientes_nr_ordenados]
    advantaged_group_nr = list_users_nr[0:15]
    disadvantaged_group_nr = list_users_nr[15:300]
    G_NR = {1: advantaged_group_nr, 2: disadvantaged_group_nr}

    print("G_RINDV")
    print(G_RINDV)

    print("G_LOSS")
    print(G_LOSS)

    print("G_NR")
    print(G_NR)

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

    avaliacoes_inicial_df.to_excel("avaliacoes_inicial.xlsx", index=False)
    avaliacoes_final_df.to_excel("avaliacoes_final.xlsx", index=False)
    recomendacoes_inicial_df.to_excel("recomendacoes_inicial.xlsx", index=False)
    recomendacoes_final_df1.to_excel("recomendacoes_final_df1.xlsx", index=False)
    recomendacoes_final_df2.to_excel("recomendacoes_final_df2.xlsx", index=False)
    recomendacoes_final_df3.to_excel("recomendacoes_final_df3.xlsx", index=False)
    recomendacoes_final_df4.to_excel("recomendacoes_final_df4.xlsx", index=False)
    recomendacoes_modelo_global_nao_federado_df.to_excel("recomendacoes_modelo_global_nao_federado_df.xlsx", index=False)

if __name__ == "__main__":
    main()
