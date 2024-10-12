# Pour démarrer l'API avec Uvicorn, il faut exécuter la commande suivante dans le terminal : uvicorn api:app --reload
# Documentation interactive (Swagger UI) : http://127.0.0.1:8000/docs

import mlflow
import mlflow.lightgbm
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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

class ClientResponse(BaseModel):
    clients_list: list

@app.post('/get_clients_list', response_model=ClientResponse)
async def get_clients_list():
    """
    Endpoint pour récupérer la liste de tous les clients.
    """
    try:
        clients_list = get_all_clients()  # Appelle la fonction définie dans utils.py
        return JSONResponse(content={"clients_list": clients_list})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/get_column_description/{column_name}")
def get_column_description(column_name: str):
    description_text = description(column_name)
    return {"column_description": description_text}



# Définition du endpoint pour récupérer les informations sur un client
@app.post('/get_client_data')
async def get_client_data(data: requestObject):
    return {"client_data" : client_info(data.client_id)}

# Définition du endpoint pour recupérer les informations générales sur le crédit demandé
@app.post('/get_credit_info')
async def get_credit_info(data: requestObject):
    return {"credit_info" : credit_info(data.client_id)}

@app.get("/get_all_features")
def get_all_features_endpoint():
    columns = get_all_features()
    return {"columns": columns}
