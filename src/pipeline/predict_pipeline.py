import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            print(data_scaled)
            preds = model.predict(data_scaled)
            print(preds)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        age: int,
        job: str,
        marital: str,
        education: str,
        default: str,
        balance: float,
        housing: str,
        loan: str,
        contact: str,
        day: int,
        month: str,
        duration: int,
        campaign: int,
        pdays: int,
        previous: int,
        poutcome: str,
    ):
        self.age = age
        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.balance = balance
        self.housing = housing
        self.loan = loan
        self.contact = contact
        self.day = day
        self.month = month
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous
        self.poutcome = poutcome
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "default": [self.default],
                "balance": [self.balance],
                "housing": [self.housing],
                "loan": [self.loan],
                "contact": [self.contact],
                "day": [self.day],
                "month": [self.month],
                "duration": [self.duration],
                "campaign": [self.campaign],
                "pdays": [self.pdays],
                "previous": [self.previous],
                "poutcome": [self.poutcome]
                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

