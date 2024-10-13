# Pour démarrer l'API avec Uvicorn, il faut exécuter la commande suivante dans le terminal : uvicorn api:app --reload
# Documentation interactive (Swagger UI) : http://127.0.0.1:8000/docs

import mlflow
import mlflow.lightgbm
from fastapi import FastAPI, HTTPException
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
    proba, prediction, features = predict(data.client_id)
    
    # Convertir numpy types en types Python natifs
    proba = float(proba)  # Conversion en float pour la sérialisation JSON
    prediction = int(prediction)  # Conversion en int pour la sérialisation JSON
    
    return {
        "result": prediction,
        "proba": proba,
        "features": features.to_dict()  # Assurez-vous que features soit sous un format sérialisable
    }



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


def encode_image_to_base64(fig):
    """Encode une figure Matplotlib en base64."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return f"data:image/png;base64,{image_base64}"


@app.post("/get_shap_waterfall_chart")
async def get_shap_waterfall_chart(client_id: float, feature_count: int = 10):
    # Utilisez la fonction predict pour obtenir les caractéristiques et la prédiction
    proba, prediction, selected_client = predict(client_id)
    
    # Vérifiez que 'selected_client' est bien formé pour SHAP
    print(f"Shape de selected_client : {selected_client.shape}")
    print(f"Colonnes de selected_client : {selected_client.columns.tolist()}")

    # Utiliser SHAP TreeExplainer pour le modèle
    explainer = shap.TreeExplainer(model)
    
    # Calculer les valeurs SHAP pour les données du client
    shap_values = explainer.shap_values(selected_client)

    # Créer le graphique waterfall
    shap.waterfall_plot(shap_values[0], max_display=feature_count)

    # Convertir le graphique en base64 pour l'affichage
    image_base64 = shap.waterfall_plot(shap_values[0], max_display=feature_count, show=False)
    
    return {"shap_chart": image_base64, "probability": proba, "prediction": prediction}

#print(df_clients.head())