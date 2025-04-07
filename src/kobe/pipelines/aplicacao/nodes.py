"""
This is a boilerplate pipeline 'aplicacao'
generated using Kedro 0.19.12
"""
import pandas as pd
import mlflow.pyfunc
import mlflow.sklearn
from sklearn.metrics import log_loss
from datetime import date


# Receber a base de teste para gerar o log_loss
#
def inferencia(raw_prod):
    print("######################## Estou rodando agora ##############")
    raw_data = raw_prod.dropna()
    prod = raw_data[['lat','lon','minutes_remaining','period','playoffs','shot_distance']]
    model_uri = "runs://uns:/4fc7f9e665f84f838e62031857300561/model"  
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    test_data = prod
    predictions = loaded_model.predict(test_data)
    predictions = pd.DataFrame(predictions, columns=["previsao"])
    
    print("######################## INICIO Log loss ##############")
    loaded_model = mlflow.sklearn.load_model(model_uri)
    pred_probs = loaded_model.predict_proba(test_data)[:, 1]
    
    y_true = raw_data['shot_made_flag']
    logloss = log_loss(y_true, pred_probs)
    print(f'pred_probs: {pred_probs}')
    print(f'Log Loss: {logloss}')
    print("######################## Termino Log loss ##############")
    
    print(predictions)
    
    run_name = f"AplicacaoDados_{date.today().strftime('%Y-%m-%d')}"
    # Log de parâmetros e métricas no MLflow
    with mlflow.start_run(run_name=run_name, nested=True):
         # Log de parâmetros
        mlflow.log_param("modelo_usado", model_uri)
        mlflow.log_param("num_amostras", len(prod))
        mlflow.log_param("colunas_usadas", list(prod.columns))
        mlflow.log_param("missing_dropped", raw_prod.isna().sum().sum())
        # Log de métricas simples
        mlflow.log_metric("Log Loss", logloss)
        mlflow.log_metric("media_previsao", predictions["previsao"].mean())
        mlflow.log_metric("classe_1", (predictions["previsao"] == 1).sum())
        mlflow.log_metric("classe_0", (predictions["previsao"] == 0).sum())
        # Log de artefato
        predictions.to_csv("predicoes.csv", index=False)
        mlflow.log_artifact("predicoes.csv")

    return predictions