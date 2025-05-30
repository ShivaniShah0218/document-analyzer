"""
    Name: model_run_test.py
    Purpose: Sending request to the summarizer model running on MLFlow
    Created At: 28th May,2025
    Author: Shivani Shah
"""
#Import necessary libraries
import requests

#Create the json request
data = {
  "dataframe_split": {
    "columns": ["text"],
    "data": [["../../DataVisualization_Article.docx"]]
  }
}

#Send the response
response = requests.post("http://127.0.0.1:5001/invocations", json=data)
print(response.json())
