"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""
from pycaret.classification import ClassificationExperiment

def treinar_dt(train_set, session_id):
    exp = ClassificationExperiment()
    exp.setup(data=train_set, target='shot_made_flag', session_id=session_id)
    #include = ["lr", "nb", "rf", "et", "knn", "dt", "svm", "dummy"]
    #model = exp.compare_models(include=include, sort="AUC")
    model = exp.create_model('dt')
    tuned_model = exp.tune_model(model, n_iter=100, optimize='AUC')
    return tuned_model

def treinar_lr(train_set, session_id):
    exp = ClassificationExperiment()
    exp.setup(data=train_set, target='shot_made_flag', session_id=session_id)
    model = exp.create_model('lr')
    tuned_model = exp.tune_model(model, n_iter=100, optimize='AUC')
    return tuned_model
