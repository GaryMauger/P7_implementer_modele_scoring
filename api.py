import mlflow
import mlflow.lightgbm
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from utils import *

# Initialiser l'application FastAPI :
# Cela crée une instance de l'application **FastAPI** appelée app, qui servira de point d'entrée pour définir et gérer les différentes routes (endpoints) de notre API.
app = FastAPI()

# Spécifier le chemin du modèle
model_path = "file:///C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/mlartifacts/950890628191069861/186b5498eff44081a9789e2d11e211ed/artifacts/model"

# Charger le modèle à partir du chemin local
model = mlflow.xgboost.load_model(model_path)

# Définition de la classe :
# Cette classe est utilisée pour modéliser les objets de requête que l'API peut recevoir, et FastAPI utilisera automatiquement cette définition pour valider les données reçues.
class requestObject(BaseModel):
    client_id: Union[float, None] = None
    feat_number : Union[int, None] = None
    feat_name : Union[str, None] = None

@app.get("/")
def great():
    return {"message": "Modèle chargé avec succès"}

@app.post('/predict')
async def predict_credit(data: requestObject):
    proba, prediction = predict(data.client_id)
    return {"result": prediction, "proba": proba}

@app.post('/get_clients_list')
async def get_clients_list():
    return {"clients_list": clients_id_list()}

# Définition du endpoint pour récupérer les informations sur un client
@app.post('/get_client_data')
async def get_client_data(data: requestObject):
    return {"client_data" : client_info(data.client_id)}

# Définition du endpoint pour recupérer les informations générales sur le crédit demandé
@app.post('/get_credit_info')
async def get_credit_info(data: requestObject):
    return {"credit_info" : credit_info(data.client_id)}
