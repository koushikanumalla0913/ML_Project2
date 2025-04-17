import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.statistical_analysis import StatisticalAnalysis
#from src.components.data_transformation import DataTransformation, DataTransformationConfig
#from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv("notebook/data/bank-full.csv",sep =";")
            logging.info(f"Columns in the dataset: {list(df.columns)}")
            logging.info("Dataset loaded successfully")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed")
            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        train_path, test_path = DataIngestion().initiate_data_ingestion()
        #logging.info(f"Type of df: {type(self.df)}")

        analysis = StatisticalAnalysis().analyze_numeric_vs_target(train_path)
    
        #train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_data, test_data)
        #print(ModelTrainer().initiate_model_trainer(train_arr, test_arr))
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
