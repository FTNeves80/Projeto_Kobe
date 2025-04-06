"""
This is a boilerplate pipeline 'aplicacao'
generated using Kedro 0.19.12
"""
import pandas as pd
import mlflow.pyfunc

def inferencia(raw_prod):
    print("######################## Estou rodando agora ##############")
    raw_data = raw_prod.dropna()
    prod = raw_data[['lat','lon','minutes_remaining','period','playoffs','shot_distance']]
    model_uri = "runs://uns:/4fc7f9e665f84f838e62031857300561/model"  
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    test_data = prod
    predictions = loaded_model.predict(test_data)
    predictions = pd.DataFrame(predictions, columns=["previsao"])
    
    print(predictions)

    return predictions