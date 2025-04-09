"""
This is a boilerplate pipeline 'aplicacao'
generated using Kedro 0.19.12
"""
import pandas as pd
import mlflow.pyfunc
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss,roc_curve, auc
import matplotlib.pyplot as plt
from datetime import date
import joblib

def inferencia(raw_prod):
    #Preparando os dados e subindo o modelo do mlflow
    raw_data = raw_prod.dropna()
    prod = raw_data[['lat','lon','minutes_remaining','period','playoffs','shot_distance']]
    model_uri = 'runs:/c37b973232204389932db377de875f41/model' #nova Decision Tree Classifier
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    test_data = prod

    #Fazendo prediçoes e tranformando resultado em DF
    predictions = loaded_model.predict(test_data)
    predictions = pd.DataFrame(predictions, columns=["previsao"])
    
    #calculando log loss
    loaded_model = mlflow.sklearn.load_model(model_uri)
    pred_probs = loaded_model.predict_proba(test_data)[:, 1]
    y_true = raw_data['shot_made_flag']
    logloss = log_loss(y_true, pred_probs)
    f1 = f1_score(y_true, predictions["previsao"])

    #Gerando curva ROC
    fpr, tpr, _ = roc_curve(y_true, pred_probs)
    roc_curve_prd = plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend()
    
    
    # Tenta obter experimento, senão cria uma novo.
    experiment_name = "AplicacaoDados"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Logando os parâmetros e métricas no MLflow
    with mlflow.start_run(run_name=f"PipelineAplicacao_{date.today().strftime('%Y-%m-%d')}",experiment_id=experiment_id,nested=True):

         # Log de parâmetros
        mlflow.log_param("modelo_usado", model_uri)
        joblib.dump(loaded_model, "modelo_aplicacao.pkl")
        mlflow.log_artifact("modelo_aplicacao.pkl", artifact_path="model")
        mlflow.log_param("num_amostras", len(prod))
        mlflow.log_param("colunas_usadas", list(prod.columns))
        mlflow.log_param("missing_dropped", raw_prod.isna().sum().sum())

        # Log de métricas simples
        mlflow.log_metric("Log Loss", logloss)
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("media_previsao", predictions["previsao"].mean())
        mlflow.log_metric("classe_1", (predictions["previsao"] == 1).sum())
        mlflow.log_metric("classe_0", (predictions["previsao"] == 0).sum())

        # Log de artefato
        predictions.to_csv("predicoes.csv", index=False)
        mlflow.log_artifact("predicoes.csv")

    return predictions,roc_curve_prd