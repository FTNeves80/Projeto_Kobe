"""
This is a boilerplate pipeline 'preparacaodados'
generated using Kedro 0.19.12
"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import ClassificationExperiment
from datetime import date

def preparacao(raw_train):
    train = raw_train[['lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_made_flag']]
    data_filtered = train.dropna()
    return data_filtered

def dividir_dados(data_filtered):
    train_set, test_set = train_test_split(data_filtered,test_size=0.2, random_state = 10)
    
    #mlflow.set_experiment("PreparacaoDados")
    
    # Cria nome com data
    run_name = f"PreparacaoDados_{date.today().strftime('%Y-%m-%d')}"
    
    # Log de parâmetros e métricas no MLflow
    with mlflow.start_run(experiment_name = "PreparacaoDados" ,run_name=run_name, nested=True):
         mlflow.log_param("split_test_size", 0.2)
         mlflow.log_param("split_random_state", 10)
         mlflow.log_metric("train_rows", train_set.shape[0])
         mlflow.log_metric("test_rows", test_set.shape[0])

    return train_set, test_set