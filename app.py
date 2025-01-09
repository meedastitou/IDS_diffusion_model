from flask import Flask, render_template, request, jsonify
import torch
import pandas as pd
from src.models.diffusion_model import DiffusionModel
from src.components.preprocess import preprocess_data, inverse_transform

app = Flask(__name__)

# Charger le modèle une seule fois au démarrage de l'application
def load_model():
    input_dim = 42  # Remplacez par la dimension réelle de vos données
    model = DiffusionModel(input_dim=input_dim)
    model.load_state_dict(torch.load("models/final/diffusion_model.pth", map_location="cpu"))
    model.eval()
    return model

# Charger le modèle et le stocker dans une variable globale
model = load_model()

# Route pour la page d'accueil
@app.route("/")
def home():
    return render_template("index.html")

# Route pour générer des données
@app.route("/generate", methods=["POST"])
def generate_data():
    # Utiliser le modèle chargé (pas besoin de le recharger)
    global model

    # Générer des données
    batch_size = int(request.form.get("batch_size", 10))  # Taille du batch (par défaut 10)
    generated_data = model.sample(batch_size=batch_size).cpu().numpy()

    # Convertir en DataFrame
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
    generated_df = pd.DataFrame(generated_data, columns=columns)

    # Appliquer l'inverse transform pour les caractéristiques continues et catégorielles
    generated_df = inverse_transform(generated_df)

    # Convertir en format HTML pour l'affichage
    data_html = generated_df.to_html(classes="table table-striped", index=False)
    
    # Renvoyer les données générées
    return jsonify({"data": data_html})

if __name__ == "__main__":
    app.run(debug=True)