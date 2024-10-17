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
model_path = "file:///C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/mlartifacts/535513794895831126/144f60891d2140538a6daad907da28a3/artifacts/model"

# Charger le modèle à partir du chemin local
model = mlflow.xgboost.load_model(model_path)

# Définition de la classe :
# Cette classe est utilisée pour modéliser les objets de requête que l'API peut recevoir, et FastAPI utilisera automatiquement cette définition pour valider les données reçues.
class requestObject(BaseModel):
    client_id: Union[float, None] = None
    feat_number : Union[int, None] = None
    feat_name : Union[str, None] = None
    seuil_nom: str = "faible"  # Valeur par défaut pour le seuil

class ClientResponse(BaseModel):
    clients_list: list


@app.get("/")
def great():
    return {"message": "Modèle chargé avec succès"}



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


@app.post('/predict')
async def predict_credit(data: requestObject):
    # Effectuer la prédiction
    proba, prediction, features, seuil_valeur, seuil_nom_affiche = predict(data.client_id, data.seuil_nom)
    
    # Convertir les types NumPy en types Python natifs pour la sérialisation JSON
    proba = float(proba)  # Conversion en float pour la sérialisation JSON
    prediction = int(prediction)  # Conversion en int pour la sérialisation JSON
    
    # Retourner les résultats
    return {
        "result": prediction,
        "proba": proba,
        "features": features.to_dict(),  # Assurez-vous que features soit sous un format sérialisable
        "threshold_value": seuil_valeur,
        "threshold_name": seuil_nom_affiche
    }


def encode_image_to_base64(fig):
    """Encode une figure Matplotlib en base64."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return f"data:image/png;base64,{image_base64}"

from fastapi.responses import StreamingResponse

@app.post("/get_shap_waterfall_chart_bis")
async def get_shap_waterfall_chart_bis(client_id: float, feature_count: int = 10):
    # Utilisez la fonction predict pour obtenir les valeurs
    try:
        proba, prediction, selected_client, seuil_valeur, seuil_nom_affiche = predict(client_id)
        
        # Convertir proba et prediction en types natifs Python
        proba = float(proba)
        prediction = float(prediction)

        # Vérifiez que 'selected_client' est bien formé pour SHAP
        print(f"Shape de selected_client : {selected_client.shape}")
        print(f"Colonnes de selected_client : {selected_client.columns.tolist()}")

        # Créer le graphique waterfall en utilisant la fonction définie dans utils
        image_base64 = shap_waterfall_chart_bis(selected_client, model, feat_number=feature_count)

        # Retourner les résultats
        return {
            "shap_chart": image_base64,
            "probability": proba,
            "prediction": prediction,
            "seuil_valeur": seuil_valeur,
            "seuil_nom_affiche": seuil_nom_affiche
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Une erreur est survenue lors de la génération du graphique SHAP.")




@app.post("/get_shap_waterfall_chart")
async def get_shap_waterfall_chart(client_id: float, feature_count: int = 10):
    # Utilisez la fonction predict pour obtenir les valeurs
    try:
        proba, prediction, selected_client, seuil_valeur, seuil_nom_affiche = predict(client_id)
        
        # Convertir proba et prediction en types natifs Python
        proba = float(proba)
        prediction = float(prediction)

        # Vérifiez que 'selected_client' est bien formé pour SHAP
        print(f"Shape de selected_client : {selected_client.shape}")
        print(f"Colonnes de selected_client : {selected_client.columns.tolist()}")

        # Créer le graphique waterfall en utilisant la fonction définie dans utils
        image_base64 = shap_waterfall_chart(selected_client, model, feat_number=feature_count)

        # Retourner les résultats
        return {
            "shap_chart": image_base64,
            "probability": proba,
            "prediction": prediction,
            "seuil_valeur": seuil_valeur,
            "seuil_nom_affiche": seuil_nom_affiche
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Une erreur est survenue lors de la génération du graphique SHAP.")
