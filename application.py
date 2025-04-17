from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)

app = application

## Route for a home page


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            age=int(request.form.get("age")),
            job=request.form.get("job"),
            marital=request.form.get("marital"),
            education=request.form.get("education"),
            default=request.form.get("default"),
            balance=float(request.form.get("balance")),
            housing=request.form.get("housing"),
            loan=request.form.get("loan"),
            contact=request.form.get("contact"),
            day=int(request.form.get("day")),
            month=request.form.get("month"),
            duration=int(request.form.get("duration")),
            campaign=int(request.form.get("campaign")),
            pdays=int(request.form.get("pdays")),
            previous=int(request.form.get("previous")),
            poutcome=request.form.get("poutcome")
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print(results)
        logging.info(f"Predictions: {results}")
        print("After Prediction")

        return render_template("home.html", results=results[0])



if __name__ == "__main__":
    app.run(host="0.0.0.0")
