import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustmeException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataTransfromartionConfigs:
    preprocess_obj_file_patrh = os.path.join("artifacts/data_transformation", "preprcessor.pkl")
    encoder_obj_file_patrh = os.path.join("artifacts/data_transformation", "all_encoder.pkl")



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransfromartionConfigs()


    def get_data_transformation_obj(self):
        try:

            logging.info(" Data Transformation Started")

            # Define the list of numerical and object features
            numerical_features = [
                'SizeOfCode', 'SizeOfInitializedData', 'BaseOfCode', 'FileAlignment',
                'MajorSubsystemVersion', 'SizeOfHeaders', 'SizeOfStackReserve', 'SectionsNb',
                'SectionsMeanEntropy', 'SectionsMinEntropy', 'SectionsMaxEntropy', 'ExportNb',
                'ResourcesNb', 'VersionInformationSize', 'Machine', 'Characteristics',
                'Subsystem', 'DllCharacteristics'
            ]

            object_features = ['protocol_type', 'service', 'flag']
            # Create pipelines for object and numerical features
            object_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("encoder", LabelEncoder())
                ]
            )

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            # Combine both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    # ("object_pipeline", object_pipeline, object_features),
                    ("numerical_pipeline", numerical_pipeline, numerical_features)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustmeException(e, sys)
        
    def inititate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path, low_memory=False)
            test_data = pd.read_csv(test_path, low_memory=False)
            logging.info("data readed it into transformation")
            
            

            logging.info("end enndoding")
            categorical_columns = ['protocol_type', 'service', 'flag']
            categorical_columns = test_data.select_dtypes(include=['object', 'category']).columns.tolist()
            # Afficher les colonnes catégorielles
            print("Colonnes catégorielles :")
            print(categorical_columns)
            logging.info("start encoding")
            print(test_data.dtypes)
            logging.info("start scaling")
            scaler = StandardScaler()
            logging.info("start scaling training data set")
            train_data = scaler.fit_transform(train_data)
            logging.info("start scaling testing data set")
            test_data = scaler.transform(test_data)
            save_object(file_path=self.data_transformation_config.preprocess_obj_file_patrh,
                        obj=scaler)
            
            return (train_data,
                    test_data,
                    self.data_transformation_config.preprocess_obj_file_patrh)



        except Exception as e:
            raise CustmeException(e, sys)
  