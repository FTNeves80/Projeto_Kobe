"""
This is a boilerplate pipeline 'preparacaodados'
generated using Kedro 0.19.12
"""
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import ClassificationExperiment

def preparacao(raw_train):
    train = raw_train[['lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_made_flag']]
    data_filtered = train.dropna()
    return data_filtered

def dividir_dados(data_filtered):
    train_set, test_set = train_test_split(data_filtered,test_size=0.2, random_state = 10)
    return train_set, test_set