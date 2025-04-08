"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""
import mlflow
import mlflow.sklearn
from pycaret.classification import ClassificationExperiment
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import date
import joblib
from sklearn.metrics import accuracy_score, f1_score, log_loss


def treinar(train_set,test_set,session_id):
    
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
    #Tunando os modelo
    tuned_model_dt = exp.tune_model(model_dt, n_iter=100, optimize='AUC')
    tuned_model_lr = exp.tune_model(model_lr, n_iter=100, optimize='AUC')

       # Testes
    data = test_set.dropna()
    X = data[['lat','lon','minutes_remaining','period','playoffs','shot_distance']]
    y = data['shot_made_flag']

    preds_dt = tuned_model_dt.predict(X)
    probas_dt = tuned_model_dt.predict_proba(X)[:, 1]

    preds_lr = tuned_model_lr.predict(X)
    probas_lr = tuned_model_lr.predict_proba(X)[:, 1]

    #mlflow.set_experiment("Treinamento")
    with mlflow.start_run(run_name=f"Teste_modelos_{date.today().strftime('%Y-%m-%d')}",nested=True):
    # Logando métricas no mesmo run
        mlflow.log_metric("dt_accuracy", accuracy_score(y, preds_dt))
        mlflow.log_metric("dt_f1", f1_score(y, preds_dt))
        mlflow.log_metric("dt_logloss", log_loss(y, probas_dt))

        mlflow.log_metric("lr_accuracy", accuracy_score(y, preds_lr))
        mlflow.log_metric("lr_f1", f1_score(y, preds_lr))
        mlflow.log_metric("lr_logloss", log_loss(y, probas_lr))

    print("Métricas logadas no MLflow com sucesso.")

    return tuned_model_dt, tuned_model_lr

