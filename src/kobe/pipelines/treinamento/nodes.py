"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""
import mlflow
import mlflow.sklearn
from pycaret.classification import ClassificationExperiment
from mlflow.tracking import MlflowClient

def treinar(train_set, session_id):

    # Renomeia o run principal criado automaticamente pelo Kedro
    client = MlflowClient()
    client.set_tag(mlflow.active_run().info.run_id, "mlflow.runName", "Modelos")

    with mlflow.start_run(run_name="Decision Tree",nested=True):
        # Setup do experimento
        exp = ClassificationExperiment()
        exp.setup(data=train_set, target='shot_made_flag', session_id=session_id)

        # Treinamento do Decision Tree
        model_dt = exp.create_model('dt')
        tuned_model_dt = exp.tune_model(model_dt, n_iter=1, optimize='AUC')
        mlflow.sklearn.log_model(tuned_model_dt, artifact_path="model_dt",registered_model_name="DecisionTree")  

    with mlflow.start_run(run_name="Logistic Regression",nested=True):
        # Treinamento do Logistic Regression
        model_lr = exp.create_model('lr')
        tuned_model_lr = exp.tune_model(model_lr, n_iter=1, optimize='AUC')
        mlflow.sklearn.log_model(tuned_model_lr, artifact_path="model_lr",registered_model_name="LogisticRegression")  

    return None