from src.logger import logging
from src.exception import CustmeException
import os, sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import torch
from torch.utils.data import TensorDataset


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustmeException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_objt:
            return pickle.load(file_objt)
    except Exception as e:
        raise CustmeException(e, sys)


def save_tensor_dataset(dataset, file_path):
    torch.save(dataset, file_path)
    print(f"Dataset sauvegardé : {file_path}")

def load_tensor_dataset(file_path):
    dataset = torch.load(file_path)
    print(f"Dataset chargé depuis : {file_path}")
    return dataset




# === 1. Fonction pour ajouter du bruit étape par étape ===
def add_progressive_noise(data, t, beta_schedule):
    """Ajoute du bruit gaussien progressivement."""
    batch_size = data.size(0)
    
    # Extraire les beta correspondants pour chaque échantillon
    beta = beta_schedule[t - 1]  # beta pour chaque échantillon (batch_size,)
    beta = beta.view(-1, 1)  # Ajuste la dimension pour correspondre à data
    
    # Générer le bruit
    noise = torch.randn_like(data) * torch.sqrt(beta)
    
    # Calculer alpha_t pour chaque échantillon
    alpha_t = torch.ones(batch_size, device=data.device)  # Initialiser alpha_t à 1
    for i in range(batch_size):
        alpha_t[i] = torch.prod(1 - beta_schedule[:t[i]])  # Produit des (1 - beta) jusqu'à t[i]
    
    alpha_t = alpha_t.view(-1, 1)  # Ajuster la dimension
    return torch.sqrt(alpha_t) * data + torch.sqrt(1 - alpha_t) * noise

# === 2. Planification des coefficients beta ===
def linear_beta_schedule(T, start=0.0001, end=0.02):
    """Renvoie une planification linéaire des coefficients beta."""
    return torch.linspace(start, end, T)
