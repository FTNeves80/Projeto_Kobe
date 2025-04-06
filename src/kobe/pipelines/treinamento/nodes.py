"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""
import mlflow
import mlflow.sklearn
from pycaret.classification import ClassificationExperiment
from mlflow.tracking import MlflowClient
import pandas 
from datetime import date

def treinar(train_set,session_id):
    
    # Setup do experimento
    exp = ClassificationExperiment()
    exp.setup(data=train_set, 
                  target='shot_made_flag', 
                  session_id=session_id, 
                  log_experiment=True, 
                  experiment_name="Treinamento",
                  experiment_custom_tags={"mlflow.runName": 
                                          f"Treinamento_Kobe_{date.today().strftime('%Y-%m-%d')}" }
                 )

    # Treinamento do Decision Tree
    model_dt = exp.create_model('dt')            
    # Treinamento do Logistic Regression
    model_lr = exp.create_model('lr')
#   with mlflow.start_run(run_name="Tuning",nested=True):
    tuned_model_dt = exp.tune_model(model_dt, n_iter=100, optimize='AUC')
    tuned_model_lr = exp.tune_model(model_lr, n_iter=100, optimize='AUC')
    return None
