import os, sys
import numpy as np
import torch
from src.logger import logging
from src.exception import CustmeException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_praparation import DataPreparation
from src.components.data_loader import DataLoaders
from src.models.diffusionmodel import DiffusionModel
from src.components.model_trainer import ModelTrainer
from src.utils import load_tensor_dataset, load_object
# from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_data, generate_data
import pandas as pd
if __name__ == "__main__":
    logging.info("start programme ")
    # data_path = "artifacts/data_ingestion/aw.gz"
    # train_model(data_path=data_path)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    generateData = generate_data(model_path="models/final/diffusion_model.pth")
    # Convert generated data back to a DataFrame
    generated_data_np = generateData.cpu().numpy()
    generated_df = pd.DataFrame(generated_data_np, columns=columns)


    scaler = load_object("artifacts/data_transformation_v2/scaler.pkl")
    continuous_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
       'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
       'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
       'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate']
    # Inverse scaling for continuous features
    generated_df[continuous_cols] = scaler.inverse_transform(generated_df[continuous_cols])
    
    encoder = load_object("artifacts/data_transformation_v2/encoder.pkl")


    categorical_cols = ['protocol_type', 'service', 'flag', 'label']

   # Décodage des caractéristiques catégorielles
    for col in categorical_cols:
        # Limiter les valeurs générées aux classes valides
        valid_classes = np.arange(len(encoder.classes_))
        generated_df[col] = np.round(generated_df[col]).astype(int)
        generated_df[col] = np.clip(generated_df[col], valid_classes.min(), valid_classes.max())
        
        # Décodage des valeurs
        generated_df[col] = encoder.inverse_transform(generated_df[col])

    # Afficher les données générées
    print(generated_df.head())


    # evaluate_data(generateData, "artifacts/data_ingestion/raw.gz")


