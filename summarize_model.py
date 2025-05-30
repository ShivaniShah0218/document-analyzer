"""
    Name: summarize_model.py
    Purpose: MLFlow Model Wrapper
    Author: Shivani Shah
    Created At: 28th May,2025
"""

#Necessary imports
import mlflow.pyfunc
from summarizer import summarize_function

class SummarizerModel(mlflow.pyfunc.PythonModel):
    """
        Custom MLFlow model class to know mlflow how to handle inference
    """
    def predict(self,context,model_input):
        """
            Purpose: method MLFlow calls when made a request to the model
            Input: Model input of dataframe
        """
        return model_input["text"].apply(summarize_function)