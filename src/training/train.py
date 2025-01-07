import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from src.models.diffusion_model import DiffusionModel
from src.components.preprocess import load_data, preprocess_data
import torch.optim as optim
from src.utils import save_object


def train_model(data_path, batch_size=128, num_epochs=50, lr=1e-3):
    model_path = os.path.join("models/final", "diffusion_model.pth")
    encoder_obj_file_path = os.path.join("artifacts/data_transformation_v2", "encoder.pkl")
    scaler_obj_file_path = os.path.join("artifacts/data_transformation_v2", "scaler.pkl")

    # Charge et prétraite les données
    df = load_data(data_path)
    df, encoder, scaler = preprocess_data(df)

    save_object(file_path=encoder_obj_file_path,
                        obj=encoder)
    save_object(file_path=scaler_obj_file_path,
                        obj=scaler)
    
    data_tensor = torch.tensor(df.values, dtype=torch.float32)

    # Crée un DataLoader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialise le modèle et l'optimiseur
    input_dim = data_tensor.shape[1]
    model = DiffusionModel(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch[0]
            noise = torch.randn_like(x)  # Ajoute du bruit
            noisy_x = x + 0.1 * noise  # Corrompt les données avec du bruit
            loss = model.training_step(noisy_x, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # save_object(model_path, model)
    # Sauvegarde le modèle
    torch.save(model.state_dict(), "models/final/diffusion_model.pth")