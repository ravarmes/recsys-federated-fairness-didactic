import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU está disponível. Modelo será treinado na GPU.")
else:
    device = torch.device("cpu")
    print("GPU não está disponível. Modelo será treinado na CPU.")