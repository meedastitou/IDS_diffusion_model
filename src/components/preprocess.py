import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(file_path):
    """Charge les données brutes."""
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
    return pd.read_csv(file_path, header=None, names=columns)

def preprocess_data(df):
    """Prétraite les données."""
    # Encode les variables catégorielles
    categorical_cols = ['protocol_type', 'service', 'flag', 'label']
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Normalise les variables continues
    continuous_cols = df.columns.drop(categorical_cols)
    scaler = MinMaxScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    return df, encoder, scaler

def save_data(df, output_path):
    """Sauvegarde les données prétraitées."""
    df.to_csv(output_path, index=False)