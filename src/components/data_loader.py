import torch
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from src.logger import logging
from src.utils import save_tensor_dataset
import os
import numpy as np
@dataclass
class DataLoaderConfigs:
    train_loader_path = os.path.join("artifacts/processed", "train_tensor_dataset.pt")
    test_loader_path = os.path.join("artifacts/processed", "test_tensor_dataset.pt")


class DataLoaders:
    def __init__(self):
        self.data_loader_config = DataLoaderConfigs()



    def to_tensor(self, X):
        logging.info("Function to tensor start")
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        elif hasattr(X, "values"):  # Cas d'un DataFrame pandas
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        else:
            raise TypeError("Le format des données n'est pas pris en charge.")
        
        logging.info("Function to tensor end")
        return X_tensor

    def get_dataloaders(self, X_train, X_test, batch_size=32):
        logging.info("function get_dataloaders start")
        # Conversion en tensors
        X_train_tensor = self.to_tensor(X_train)
        X_test_tensor = self.to_tensor(X_test)

        # Créer les datasets
        train_dataset = TensorDataset(X_train_tensor)
        test_dataset = TensorDataset(X_test_tensor)

        # Créer les DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        
        logging.info("function get_dataloaders end")
        save_tensor_dataset(train_loader, self.data_loader_config.train_loader_path)
        save_tensor_dataset(test_loader, self.data_loader_config.test_loader_path)
        return (train_loader, test_loader, self.data_loader_config)
