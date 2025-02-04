import torch
import pandas as pd
from src.models.diffusion_model import DiffusionModel
from src.components.preprocess import load_data, preprocess_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data(model_path, batch_size=128, num_steps=1000):
    # Charge le modèle
    input_dim = 42  # Remplacez par la dimension réelle de vos données
    model = DiffusionModel(input_dim=input_dim).to(device)  # Déplacer sur le GPU
    model.load_state_dict(torch.load(model_path, map_location=device))  # Charger sur le GPU
    model.eval()

    # Génère de nouvelles données
    generated_data = model.sample(batch_size=batch_size, num_steps=num_steps).cpu()
    return generated_data

def evaluate_data(generated_data, original_data_path):
    # Charge les données originales
    df = load_data(original_data_path)
    df, _ , _ = preprocess_data(df)

    # Compare les statistiques
    print("Original Data Mean:\n", df.mean())
    print("Generated Data Mean:\n", generated_data.mean())

    print("Original Data Std:\n", df.std())
    print("Generated Data Std:\n", generated_data.std())