import pandas as pd
import os
from src.logger import logging 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder

col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
             "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
             "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
             "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
             "dst_host_srv_rerror_rate","label"]

columns_to_drop = {'count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'num_compromised', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate', 'same_srv_rate', 'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate', 'srv_count', 'srv_rerror_rate', 'dst_host_srv_count', 'dst_host_serror_rate', 'num_root', 'dst_host_rerror_rate'}
columns_use= ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'root_shell', 'su_attempted',
       'num_file_creations', 'num_shells', 'num_access_files',
       'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'diff_srv_rate',
       'srv_diff_host_rate', 'dst_host_count', 'dst_host_diff_srv_rate',
       'dst_host_srv_diff_host_rate']

@dataclass
class DataPreparationConfig:
    train_data_path = os.path.join("artifacts/processed", "train.gz")
    test_data_path = os.path.join("artifacts/processed", "test.gz")
    raw_data_path = os.path.join("artifacts/processed", "processed_data.gz")
    encoder_obj_file_patrh = os.path.join("artifacts/data_transformation", "all_encoder.pkl")


class DataPreparation:
    def __init__(self, row_path):
        self.row_path = row_path
        self.preparationconfig = DataPreparationConfig()
        self.col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
             "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
             "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
             "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
             "dst_host_srv_rerror_rate","label"]

        self.columns_to_drop = ['count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'num_compromised', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate', 'same_srv_rate', 'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate', 'srv_count', 'srv_rerror_rate', 'dst_host_srv_count', 'dst_host_serror_rate', 'num_root', 'dst_host_rerror_rate']

        self.columns_use= ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'root_shell', 'su_attempted',
            'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_diff_srv_rate',
            'dst_host_srv_diff_host_rate', 'label']

    def load_data(self, file_path):
        return pd.read_csv(file_path, names=self.col_names,usecols=self.columns_use, low_memory=False)

    # def drop_columns(self, df):
    #     df2 = df.drop(columns=list(self.columns_to_drop))
    #     return df2
    
    def preprocess_data(self, df):

        missing_values = df.isnull().sum()
        logging.info("Valeurs manquantes par colonne :")
        logging.info(missing_values[missing_values > 0])
        data_cleaned = df.dropna()
        logging.info("End preprcess_data")
        return data_cleaned
    def encoding_data(self, df):
        categorical_columns = ['protocol_type', 'service', 'flag', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_diff_srv_rate', 'dst_host_srv_diff_host_rate']
            # Dictionnaire pour stocker les encodeurs de chaque colonne
        label_encoders = {}

        for col in categorical_columns:
            logging.info(f"Encodage de la colonne d'entraînement : {col}")
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            label_encoders[col] = label_encoder  # Sauvegarder l'encodeur


        df["label"] = df["label"].apply(lambda x: 1 if x == 'normal.' else 0)

        save_object(file_path=self.preparationconfig.encoder_obj_file_patrh,
                        obj=label_encoders)
        
        return df
    def split_data(self, df):
        train_set, test_set = train_test_split(df, test_size = .20, random_state=42)
        logging.info("Data spliteted into train and test")
        return (train_set, test_set)

    def save_data(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info("data cleaned and save it")

    def prapation_data(self):
        raw_data = self.load_data(self.row_path)
        # row_data = self.drop_columns(raw_data)
        processed_data = self.preprocess_data(raw_data)
        endoded_data = self.encoding_data(processed_data)
        train_set, test_set = self.split_data(endoded_data)

        self.save_data(processed_data, self.preparationconfig.raw_data_path)
        self.save_data(train_set, self.preparationconfig.train_data_path)
        self.save_data(test_set, self.preparationconfig.test_data_path)
        logging.info("end data preparation")
        return (self.preparationconfig.train_data_path, self.preparationconfig.test_data_path)

if __name__ == "__main__":
    obj = DataPreparation(row_path="notbook/data/kddcup.data.gz")
    obj.prapation_data()
    logging.info("end data preparation")

