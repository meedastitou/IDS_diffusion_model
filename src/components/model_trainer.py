import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
import torch
import torch.nn.init as init
from src.logger import logging
from src.exception import CustmeException
from src.utils import add_progressive_noise, add_progressive_noise, linear_beta_schedule
import os, sys
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    train_model_file_apth = os.path.join("artifacts/model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self, model, train_loader, test_loader, device="cpu"):
        try:
            T = 30  # Nombre total d'étapes de bruit
            beta_schedule = linear_beta_schedule(T).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(10):
                for batch in tqdm(train_loader):
                    batch_data = batch[0]
                    
                    # Ajouter du bruit aux données
                    noise = torch.randn_like(batch_data) * 0.1
                    noisy_data = batch_data + noise
                    
                    # Prédire les données originales
                    predicted_data = model(noisy_data)
                    
                    # Calculer la perte
                    loss = criterion(predicted_data, batch_data)
                    
                    # Mise à jour du modèle
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')

        except Exception as e:
            raise CustmeException(e, sys)