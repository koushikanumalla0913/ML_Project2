import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_cols = [
                "age",
                "balance",
                "day",
                "duration",
                "campaign",
                "pdays",
                "previous",
            ]
            cat_cols = [
                "job",
                "marital",
                "default",
                "housing",
                "loan",
                "contact",
                "month",
                "poutcome"
            ]
                 
            

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            logging.info(f"Numerical columns: {num_cols}")
            logging.info(f"Categorical columns: {cat_cols}")

            return ColumnTransformer([
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test datasets")

            preprocessor = self.get_data_transformer_object()
            target = "y"

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]
            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            logging.info("Applying preprocessing pipeline")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logging.info("Saved preprocessor object")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
