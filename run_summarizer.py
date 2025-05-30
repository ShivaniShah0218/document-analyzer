"""
    Name: run_summarizer.py
    Purpose: Entry point script for running summarizer as part of the MLFlow project
    Author: Shivani Shah
    Created At: 28th May,2025
"""
#Import necessary libraries and classes
import pandas as pd
import mlflow
from summarize_model import SummarizerModel

model_path="summarize_model"


#Logging the model
with mlflow.start_run():
    mlflow.pyfunc.log_model(artifact_path=model_path, python_model=SummarizerModel())