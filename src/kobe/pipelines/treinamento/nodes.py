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

def treinar(train_set,test_set, session_id):

    # Renomeia o run principal criado automaticamente pelo Kedro
    client = MlflowClient()
    client.set_tag(mlflow.active_run().info.run_id, "mlflow.runName", "Modelos")

    with mlflow.start_run(run_name="Decision Tree",nested=True):
        # Setup do experimento
        exp = ClassificationExperiment()
        exp.setup(data=train_set, target='shot_made_flag', session_id=session_id)
                
        # Treinamento do Decision Tree
        model_dt = exp.create_model('dt')
##############  Salvando o relató em excel #############################
        results_df = exp.pull()
        model_name = "teste"  # personalize com o modelo real
        today = date.today().isoformat()
        filename = f"data/08_reporting/novo_{model_name}_{today}.xlsx"
        results_df.to_excel(filename)
########################################################################
        
        tuned_model_dt = exp.tune_model(model_dt, n_iter=1000, optimize='AUC')
        mlflow.sklearn.log_model(tuned_model_dt, 
                                 artifact_path="model_dt",
                                 registered_model_name="DecisionTree")
        print("########################################################################") 
        print("Tamanho do train_set:", len(train_set)) 
        print("Tamanho do test_set:", len(test_set))  
        mlflow.log_metric("train_size", len(train_set))
        mlflow.log_metric("train_percent", 80)
        mlflow.log_metric("test_size", len(test_set))
        mlflow.log_metric("test_percent", 20)
        print("########################################################################") 

    with mlflow.start_run(run_name="Logistic Regression",nested=True):
        # Treinamento do Logistic Regression
        model_lr = exp.create_model('lr')
        tuned_model_lr = exp.tune_model(model_lr, n_iter=1000, optimize='AUC')

        
        mlflow.sklearn.log_model(tuned_model_lr, 
                                 artifact_path="model_lr",
                                 registered_model_name="LogisticRegression")  
        print("########################################################################") 
        print("Tamanho do train_set:", len(train_set)) 
        print("Tamanho do test_set:", len(test_set))  
        mlflow.log_metric("train_size", len(train_set))
        mlflow.log_metric("train_percent", 80)
        mlflow.log_metric("test_size", len(test_set))
        mlflow.log_metric("test_percent", 20)
        print("########################################################################") 

    return None