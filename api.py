###################################################
###################################################       IMPORTATIONS       ######################################################
###################################################

from fastapi import FastAPI, HTTPException
import mlflow.xgboost
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional, List
from sklearn.preprocessing import MinMaxScaler
import pipeline_features_eng
from enum import Enum
import shap
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Ajoutez ceci au début de votre fichier


###################################################
###################################################       CRÉATION DE L'APPLICATION FASTAPI       ######################################################
###################################################

app = FastAPI()


###################################################
###################################################       CHARGEMENT DES DONNÉES ET DU MODÈLE       ######################################################
###################################################

# Charger le modèle
model_path = "file:///C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/mlartifacts/535513794895831126/144f60891d2140538a6daad907da28a3/artifacts/model"
model = mlflow.xgboost.load_model(model_path)

# Charger les données au démarrage
data_path = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/"
df_clients = pipeline_features_eng.execute_pipeline()
df_application_test = pd.read_csv(data_path + 'Projet+Mise+en+prod+-+home-credit-default-risk/' + 'application_test.csv')

df_clients.head()
df_application_test.head()

# Filtrer les clients
df_clients = df_clients[df_clients['SK_ID_CURR'].isin(df_application_test['SK_ID_CURR'])]

# 1. Convertir SK_ID_CURR en entier (si ce n'est pas déjà fait)
df_clients['SK_ID_CURR'] = df_clients['SK_ID_CURR'].astype(int)

# 2. Supprimer la colonne TARGET, car elle n'est pas nécessaire pour les prédictions
# Si tu veux conserver une copie de df_clients sans TARGET, fais-le avant de supprimer.
df_clients.drop(columns='TARGET', inplace=True)

# 3. Définir SK_ID_CURR comme index du DataFrame
# Cela permettra un accès facile aux données d'un client spécifique par son ID
df_clients.set_index('SK_ID_CURR', inplace=True)


# 1. Créer une instance de MinMaxScaler
# La plage est définie de 0 à 1
scaler = MinMaxScaler(feature_range=(0, 1))

# 2. Appliquer le scaler sur df_clients
# Le scaler ne doit pas prendre en compte l'index
df_clients_scaled = scaler.fit_transform(df_clients)

# 3. Convertir le résultat en DataFrame
# Cela permet de conserver les noms des colonnes d'origine
df_clients_scaled = pd.DataFrame(df_clients_scaled, columns=df_clients.columns, index=df_clients.index)
df_clients_scaled.head()


###################################################
###################################################       MODÈLES DE DONNÉES       ######################################################
###################################################

# Définition des seuils avec leurs noms associés du business score
thresholds = {
    "Sans": {"valeur": 0.05, "nom": "Sans risque"},
    "Faible": {"valeur": 0.17, "nom": "Faible coût"},
    "Modéré": {"valeur": 0.50, "nom": "Coût modéré"},
    "Elevé": {"valeur": 0.28, "nom": "Coût élevé"}
}

class SeuilNom(str, Enum):
    Sans = "Sans"
    Faible = "Faible"
    Modéré = "Modéré"
    Elevé = "Elevé"

class RequestObject(BaseModel):
    client_id: Optional[int] = None
    feat_number: Optional[int] = None
    feat_name: Optional[str] = None
    seuil_nom: SeuilNom = SeuilNom.Faible  # Utilise l'Enum avec une valeur par défaut

# Modèle de réponse
class ClientResponse(BaseModel):
    probability: float
    prediction: int
    features: dict
    seuil_value: float
    seuil_name: str

class ClientList(BaseModel):
    clients_list: List[str]  # Liste de chaînes de caractères



###################################################       ROUTE D'ACCUEIL       ######################################################

# Route d'accueil
@app.get("/")
def read_root():
    return {"message": "API de prédiction avec FastAPI"}



###################################################
###################################################       FONCTION DE PRÉDICTION       ######################################################
###################################################

def predict(client_id, seuil_nom="Faible"):
    """
    Effectue la prédiction du score de crédit pour un client donné en utilisant le seuil spécifié.
    
    Args:
        client_id (int): L'identifiant du client pour lequel la prédiction est effectuée.
        seuil_nom (str): Le nom du seuil à utiliser pour la prédiction (défaut : "faible").
    
    Returns:
        tuple: Contient la probabilité, la classe prédite (0 ou 1), les features utilisées, le seuil et son nom.
    """
    
    # Récupérer le client sélectionné dans le DataFrame
    selected_client = df_clients_scaled.loc[df_clients_scaled.index == client_id]
    
    # Vérifiez si 'selected_client' contient un client
    if selected_client.empty:
        raise ValueError(f"Aucun client trouvé avec SK_ID_CURR = {client_id}")
    
    # Supprimer la colonne SK_ID_CURR si elle est présente dans le DataFrame
    client_features = selected_client.drop(columns=['TARGET'], errors='ignore')  # Ignore si TARGET n'est pas présent
    
    # Calcul de la probabilité de la classe positive
    proba = model.predict_proba(client_features)[0][1]  # Probabilité de défaut (classe positive)
    
    # Récupérer le seuil correspondant et son nom
    seuil_info = thresholds[seuil_nom]
    seuil_valeur = seuil_info["valeur"]
    seuil_nom_affiche = seuil_info["nom"]
    
    # Faire la prédiction en fonction du seuil
    prediction = int(proba >= seuil_valeur)  # Prédiction basée sur le seuil choisi (0 ou 1)
    
    # Retourner la probabilité, la prédiction, les features utilisées, le seuil avec sa valeur et son nom
    return proba, prediction, client_features, seuil_valeur, seuil_nom_affiche



###################################################       ROUTE POUR LA PRÉDICTION       ######################################################

@app.post("/predict", response_model=ClientResponse)
def predict_client(request: RequestObject):
    if request.client_id is None:
        raise HTTPException(status_code=400, detail="client_id est requis")
    try:
        # Appel de la fonction de prédiction
        proba, prediction, features, seuil_valeur, seuil_nom_affiche = predict(request.client_id, request.seuil_nom)
        
        # Construction de la réponse
        response = ClientResponse(
            probability=proba,
            prediction=prediction,
            features=features.to_dict(orient='records')[0],  # Convertir en dict
            seuil_value=seuil_valeur,
            seuil_name=seuil_nom_affiche
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))



###################################################
###################################################       FONCTION POUR GÉNÉRER LE GRAPHIQUE WATERFALL       ######################################################
###################################################

def plot_waterfall(client_id):
    # Récupérer le client sélectionné dans le DataFrame
    selected_client = df_clients_scaled.loc[df_clients_scaled.index == client_id]

    if selected_client.empty:
        raise ValueError(f"Aucun client trouvé avec SK_ID_CURR = {client_id}")

    # Supprimer la colonne TARGET si elle est présente dans le DataFrame
    client_features = selected_client.drop(columns=['TARGET'], errors='ignore')  # Ignore si TARGET n'est pas présent

    # Calculer les valeurs SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(client_features)

    # Extraire les 10 features les plus importantes
    shap_values_client = shap_values.values[0]  # SHAP values pour le client
    feature_names = client_features.columns
    indices = np.argsort(np.abs(shap_values_client))[-10:]  # Indices des 10 plus grands en valeur absolue
    top_features = feature_names[indices]
    top_shap_values = shap_values_client[indices]

    # Afficher le graphique waterfall
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_values[0], show=False)  # Utiliser shap_values pour le client
    plt.title(f"Graphique Waterfall pour le client {client_id}")
    
    plt.tight_layout()
    
    # Enregistrer le graphique en tant qu'image et l'encoder en base64 pour l'affichage
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # Fermer la figure pour éviter l'affichage dans le notebook
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{image_base64}"  # Retourne l'image en base64 pour l'affichage


###################################################       ROUTE POUR LE GRAPHIQUE WATERFALL       ######################################################

@app.get("/waterfall/{client_id}")
def get_waterfall(client_id: int):
    try:
        image_base64 = plot_waterfall(client_id)
        return {"waterfall_image": image_base64}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

 
###################################################
###################################################       FONCTION POUR LES INFORMATIONS CLIENTS       ######################################################
###################################################
    
###################################################       ROUTE POUR LA LISTE DES CLIENTS       ######################################################

@app.get("/clients")
def get_clients():
    """Renvoie une liste d'identifiants clients."""
    client_ids = df_clients_scaled.index.tolist()  # Récupérer les IDs clients
    return {"client_ids": client_ids}


###################################################       FONCTION POUR DES INFORMATIONS SUR LE CLIENT       ######################################################

def client_info(client_id):
    """
    Retourne les informations personnelles du client sélectionné.
    """
    client_info_columns = [
        "CODE_GENDER", "CNT_CHILDREN", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "AMT_INCOME_TOTAL"
    ]
    
    client_data = df_application_test.loc[df_application_test['SK_ID_CURR'] == client_id, client_info_columns]
    
    if client_data.empty:
        raise HTTPException(status_code=404, detail=f"Aucun client trouvé avec SK_ID_CURR = {client_id}")

    client_info = client_data.T.fillna('N/A')  # Transposer et remplacer les valeurs manquantes par 'N/A'
    return client_info.to_dict()  # Convertir en dictionnaire pour un format JSON


###################################################       ROUTE POUR DES INFORMATIONS SUR LE CLIENT       ######################################################

@app.get("/client_info/{client_id}")
def get_client_info(client_id: int):
    """
    Endpoint pour obtenir les informations d'un client donné.
    """
    return client_info(client_id)


###################################################       FONCTION POUR DES INFORMATIONS SUR LE CREDIT       ######################################################

def credit_info(client_id):
    """
    Retourne les informations de crédit pour le client sélectionné.
    """
    credit_info_columns = ["NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    credit_data = df_application_test.loc[df_application_test['SK_ID_CURR'] == client_id, credit_info_columns]

    if credit_data.empty:
        raise HTTPException(status_code=404, detail=f"Aucune information de crédit trouvée pour SK_ID_CURR = {client_id}")
    credit_info = credit_data.T.fillna('N/A')
    return credit_info.to_dict()  # Transpose et convertit en dictionnaire


###################################################       ROUTE POUR DES INFORMATIONS SUR LE CREDIT       ######################################################

@app.get("/credit_info/{client_id}")
def get_credit_info(client_id: int):
    """
    Endpoint pour obtenir les informations de crédit d'un client donné.
    """
    return credit_info(client_id)

