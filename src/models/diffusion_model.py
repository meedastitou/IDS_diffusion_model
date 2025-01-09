import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, x, noise):
        predicted_noise = self(x)
        loss = nn.MSELoss()(predicted_noise, noise)
        return loss

    def sample(self, batch_size, num_steps=1000):
        with torch.no_grad():
            # Commence avec du bruit aléatoire
            x = torch.randn(batch_size, self.net[-1].out_features, device=next(self.parameters()).device)
            for _ in range(num_steps):
                predicted_noise = self(x)
                x = x - 0.01 * predicted_noise  # Étape de diffusion inverse
            return x