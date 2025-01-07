import os, sys
import torch
from src.logger import logging
from src.exception import CustmeException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_praparation import DataPreparation
from src.components.data_loader import DataLoaders
from src.models.diffusionmodel import DiffusionModel
from src.components.model_trainer import ModelTrainer
from src.utils import load_tensor_dataset
# from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass

if __name__ == "__main__":
    logging.info("start programme ")

    # logging.info("start data ingestion")
    # obj = DataIngestion()
    # row_path = obj.inititate_data_ingestion()
    # logging.info("end data ingestion")
    
    # row_path = "artifacts/processed/processed_data.gz"

    # logging.info("start data preparation")
    # preparation = DataPreparation(row_path)
    # train_data_path, test_data_path = preparation.prapation_data()
    # logging.info("end data preparation")

    # train_data_path, test_data_path = "artifacts/processed/train.gz", "artifacts/processed/test.gz"
    # logging.info("start data transformation")
    # data_transformation = DataTransformation()
    # train_set, test_set, _ = data_transformation.inititate_data_transformation(train_data_path, test_data_path)
    # logging.info("end data transformation")
    

    # logging.info("start data loader")
    # data_loader = DataLoaders()
    # train_loader, test_loader, path = data_loader.get_dataloaders(X_train= train_set, X_test=test_set)
   
    # print(path.test_loader_path)
    # print(path.train_loader_path)
    train_loader = load_tensor_dataset("artifacts/processed/train_tensor_dataset.pt")
    test_loader = load_tensor_dataset("artifacts/processed/test_tensor_dataset.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")
    model = DiffusionModel(input_dim=26, hidden_dim=128, output_dim=26)


    
    model_training = ModelTrainer()
    model_training.inititate_model_trainer(model, train_loader, test_loader, device=device)

    # src\pipeline\training_pipeline.py