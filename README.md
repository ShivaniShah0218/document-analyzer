# document-analyser
End-to-end document analyser system generating summary from the document. In this project, the summarizer model is deployed on MLFlow, and is running on CUDA 11.6

## Table of Contents
- Technologies Used
- Installation
- Usage

## Technologies Used
- OS: Windows 10
- Python 3.8.8
- MLFlow
- langchain
- transformers
- pytorch
- conda

## Usage
1.  Run the command mlflow run . -P input_file="<Path of the file>" . It will have the runid generated
2.  mlflow models serve -m runs:/<Run id generated in above command>/summarize_model -p 5001 --no-conda
3. From another command prompt, run the model_run_test file by passing the file path

