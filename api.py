###################################################
###################################################       IMPORTATIONS       ######################################################
###################################################

from fastapi import FastAPI, HTTPException
import mlflow.xgboost
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional, List
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
#import pipeline_features_eng
from enum import Enum
import shap
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import os
matplotlib.use('Agg')  # Ajoutez ceci au début de votre fichier


###################################################
###################################################       CRÉATION DE L'APPLICATION FASTAPI       ######################################################
###################################################

app = FastAPI()


###################################################
###################################################       CHARGEMENT DES DONNÉES ET DU MODÈLE       ######################################################
###################################################

##################################################

model_path = "./Data/model"
model = mlflow.xgboost.load_model(model_path)

data_path = "./Data/"

# Charger les fichiers CSV
df_application_test = pd.read_csv(os.path.join(data_path, 'df_application_test.csv'))
df_clients = pd.read_csv(os.path.join(data_path, 'df_clients.csv'))

# Charger le fichier des descriptions de colonnes
column_description = pd.read_csv(os.path.join(data_path, "HomeCredit_columns_description.csv"), 
                                 usecols=['Row', 'Description'], 
                                 index_col=0, 
                                 encoding='unicode_escape')


##################################################

# Filtrer les clients
#df_clients = df_clients[df_clients['SK_ID_CURR'].isin(df_application_test['SK_ID_CURR'])]

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
    "Faible": {"valeur": 0.05, "nom": "Très faible risque de refus"},
    "Modéré": {"valeur": 0.17, "nom": "Risque modéré de refus"},
    "Neutre": {"valeur": 0.50, "nom": "Risque neutre de refus"},
    "Elevé": {"valeur": 0.70, "nom": "Risque élevé de refus"}
}

class SeuilNom(str, Enum):
    Faible = "Faible"
    Modéré = "Modéré"
    Neutre = "Neutre"
    Elevé = "Elevé"

class RequestObject(BaseModel):
    client_id: Optional[int] = None
    feat_number: Optional[int] = None
    feat_name: Optional[str] = None
    seuil_nom: SeuilNom = SeuilNom.Modéré  # Utilise l'Enum avec une valeur par défaut

# Modèle de réponse
class ClientResponse(BaseModel):
    probability: float
    prediction: int
    features: dict
    seuil_value: float
    seuil_name: str

class ClientList(BaseModel):
    clients_list: List[str]  # Liste de chaînes de caractères

class ClientComparisonRequest(BaseModel):
    client_id: int
    variables: list[str]



###################################################       ROUTE D'ACCUEIL       ######################################################

# Route d'accueil
@app.get("/")
def read_root():
    return {"message": "API de prédiction avec FastAPI"}



###################################################
###################################################       FONCTION DE PRÉDICTION       ######################################################
###################################################

def predict(client_id, seuil_nom="Modéré"):
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
    
    # Réduire la taille du texte des axes
    plt.gca().tick_params(axis='x', labelsize=8)
    plt.gca().tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    
    # Enregistrer le graphique en tant qu'image et l'encoder en base64 pour l'affichage
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # Fermer la figure pour éviter l'affichage dans le notebook
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{image_base64}"  # Retourne l'image en base64 pour l'affichage


def plot_global_feature_importance(df_clients_scaled):
    """
    Fonction pour calculer et afficher l'importance des caractéristiques globales avec SHAP.

    Parameters:
    - X : DataFrame contenant les caractéristiques du jeu de données.

    Returns:
    - str: Image en base64 de l'importance des caractéristiques globales.
    """
    # Calculer les valeurs SHAP pour l'ensemble du jeu de données
    explainer = shap.Explainer(model, df_clients_scaled)  # Utiliser le modèle et les caractéristiques
    shap_values = explainer(df_clients_scaled)

    # Calculer l'importance globale des caractéristiques
    feature_importance = np.abs(shap_values.values).mean(axis=0)  # Moyenne des valeurs absolues

    # Créer un DataFrame pour l'importance des caractéristiques
    feature_importance_df = pd.DataFrame({
        'Feature': df_clients_scaled.columns,
        'Importance': feature_importance
    })

    # Trier par importance décroissante
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Afficher le graphique de l'importance des caractéristiques
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10], color='skyblue')
    plt.xlabel('Importance moyenne (valeurs SHAP)')
    plt.title('Importance des caractéristiques globales')
    plt.gca().invert_yaxis()  # Inverser l'axe y pour afficher la plus importante en haut

    # Enregistrer le graphique en tant qu'image et l'encoder en base64 pour l'affichage
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # Fermer la figure pour éviter l'affichage dans le notebook
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{image_base64}"  # Retourne l'image en base64 pour l'affichage


def get_top_positive_negative_features(client_id):
    # Récupérer le client sélectionné dans le DataFrame
    selected_client = df_clients_scaled.loc[df_clients_scaled.index == client_id]

    if selected_client.empty:
        raise ValueError(f"Aucun client trouvé avec SK_ID_CURR = {client_id}")

    # Supprimer la colonne TARGET si elle est présente dans le DataFrame
    client_features = selected_client.drop(columns=['TARGET'], errors='ignore')  # Ignore si TARGET n'est pas présent

    # Calculer les valeurs SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(client_features)

    # Extraire les SHAP values pour le client
    shap_values_client = shap_values.values[0]  # SHAP values pour le client
    feature_names = client_features.columns

    # Organiser les données dans un DataFrame pour faciliter le tri
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values_client
    })

    # Trier par ordre décroissant pour les valeurs positives et croissant pour les négatives
    top_positive = shap_df[shap_df["SHAP Value"] > 0].sort_values(by="SHAP Value", ascending=False).head(10)
    top_negative = shap_df[shap_df["SHAP Value"] < 0].sort_values(by="SHAP Value", ascending=True).head(10)

    # Retourner les résultats sous forme de dictionnaires pour l'API ou utilisation directe
    return {
        "top_positive": top_positive.to_dict(orient="records"),
        "top_negative": top_negative.to_dict(orient="records")
    }


###################################################       ROUTE POUR LE GRAPHIQUE WATERFALL       ######################################################

@app.get("/waterfall/{client_id}")
def get_waterfall(client_id: int):
    try:
        image_base64 = plot_waterfall(client_id)
        return {"waterfall_image": image_base64}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

# Exemple d'utilisation de la fonction dans votre endpoint
@app.get("/global_feature_importance/")
def get_global_feature_importance():
    # Supposons que df_clients_scaled soit votre DataFrame contenant les données des clients
    global_feature_importance_image = plot_global_feature_importance(df_clients_scaled.drop(columns=['TARGET'], errors='ignore'))
    return {"image": global_feature_importance_image}


@app.get("/top_features/{client_id}")
def get_top_features(client_id: int):
    try:
        top_features = get_top_positive_negative_features(client_id)
        return top_features
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


 
############################################################################################################################################################
###################################################       FONCTION POUR LES INFORMATIONS CLIENTS       ######################################################
############################################################################################################################################################
    
###################################################       ROUTE POUR LA LISTE DES CLIENTS       ######################################################

@app.get("/clients")
def get_clients():
    """Renvoie une liste d'identifiants clients."""
    client_ids = df_clients_scaled.index.tolist()  # Récupérer les IDs clients
    return {"client_ids": client_ids}


###################################################
###################################################       FONCTION POUR DES INFORMATIONS SUR LE CLIENT       ######################################################
###################################################


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


@app.post("/client_position")
def get_client_position(request: ClientComparisonRequest):
    """
    Retourne la comparaison entre un client, la moyenne, et la médiane pour les variables sélectionnées.
    """
    # Étape 1 : Vérifier si le client existe
    client_data = df_application_test.loc[df_application_test['SK_ID_CURR'] == request.client_id]
    if client_data.empty:
        raise HTTPException(status_code=404, detail=f"Aucun client trouvé avec SK_ID_CURR = {request.client_id}")

    # Étape 2 : Vérifier les variables demandées
    for variable in request.variables:
        if variable not in df_application_test.columns:
            raise HTTPException(status_code=400, detail=f"Variable {variable} introuvable")

    # Étape 3 : Calculer les comparaisons pour chaque variable
    comparison_results = []
    for variable in request.variables:
        client_value = client_data[variable].values[0]  # Récupérer la valeur du client
        global_mean = df_application_test[variable].mean()  # Moyenne globale
        global_median = df_application_test[variable].median()  # Médiane globale

        # Assurer la sérialisation en types natifs
        client_value = float(client_value) if pd.notna(client_value) else None
        global_mean = float(global_mean) if pd.notna(global_mean) else None
        global_median = float(global_median) if pd.notna(global_median) else None

        comparison_results.append({
            "variable": variable,
            "client_value": client_value,
            "global_mean": global_mean,
            "global_median": global_median,
            "difference_mean": client_value - global_mean if client_value is not None and global_mean is not None else None,
            "difference_median": client_value - global_median if client_value is not None and global_median is not None else None
        })

    # Étape 4 : Retourner les résultats
    return {
        "client_id": request.client_id,
        "comparisons": comparison_results
    }


@app.get("/available_variables/{client_id}")
def get_available_variables(client_id: int):
    """
    Retourne la liste de toutes les colonnes disponibles dans le DataFrame df_application_test
    pour lesquelles le client possède des données non manquantes, triées par ordre alphabétique,
    ainsi que leur type (numérique ou catégoriel).
    """
    # Vérifier si le DataFrame existe
    if df_application_test is not None:
        # Filtrer le client spécifique
        client_data = df_application_test.loc[df_application_test['SK_ID_CURR'] == client_id]

        # Vérifier si le client existe dans le DataFrame
        if client_data.empty:
            raise HTTPException(status_code=404, detail=f"Aucun client trouvé avec SK_ID_CURR = {client_id}")

        # Sélectionner toutes les colonnes
        all_columns = df_application_test.columns

        # Filtrer les colonnes où le client a des données non manquantes
        available_columns = [
            col for col in all_columns if not pd.isna(client_data[col].values[0])
        ]

        # Déterminer le type de chaque colonne (numérique ou catégorielle)
        variable_info = [
            {
                "name": col,
                "type": "numeric" if pd.api.types.is_numeric_dtype(df_application_test[col]) else "categorical"
            }
            for col in available_columns
        ]

        # Trier par ordre alphabétique des noms de colonnes
        variable_info = sorted(variable_info, key=lambda x: x["name"])

        return {"variables": variable_info}

    else:
        raise HTTPException(status_code=500, detail="Le DataFrame df_application_test est introuvable ou vide.")








@app.post("/client_distribution")
def get_client_distribution(request: ClientComparisonRequest):
    """
    Retourne les informations sur la position du client et la distribution pour les variables sélectionnées,
    incluant les variables catégorielles.
    """
    # Étape 1 : Vérifier si le client existe
    client_data = df_application_test.loc[df_application_test['SK_ID_CURR'] == request.client_id]
    if client_data.empty:
        raise HTTPException(status_code=404, detail=f"Aucun client trouvé avec SK_ID_CURR = {request.client_id}")

    # Étape 2 : Vérifier les variables demandées
    for variable in request.variables:
        if variable not in df_application_test.columns:
            raise HTTPException(status_code=400, detail=f"Variable {variable} introuvable")

    # Étape 3 : Préparer les résultats pour chaque variable
    distribution_results = []
    for variable in request.variables:
        try:
            # Extraire les valeurs non manquantes pour la distribution
            non_missing_values = df_application_test[variable].dropna()

            # Récupérer la valeur du client
            client_value = client_data[variable].values[0]

            # Si la variable est numérique
            if pd.api.types.is_numeric_dtype(df_application_test[variable]):
                global_mean = (
                    float(non_missing_values.mean()) if not non_missing_values.empty else None
                )
                global_median = (
                    float(non_missing_values.median()) if not non_missing_values.empty else None
                )
                global_min = (
                    float(non_missing_values.min()) if not non_missing_values.empty else None
                )
                global_max = (
                    float(non_missing_values.max()) if not non_missing_values.empty else None
                )
                global_std = (
                    float(non_missing_values.std()) if not non_missing_values.empty else None
                )
                global_values = [float(val) for val in non_missing_values]

                distribution_results.append({
                    "variable": variable,
                    "type": "numeric",
                    "client_value": float(client_value) if pd.notna(client_value) else None,
                    "global_mean": global_mean,
                    "global_median": global_median,
                    "global_min": global_min,
                    "global_max": global_max,
                    "global_std": global_std,
                    "global_values": global_values,
                    "difference_mean": (
                        float(client_value) - global_mean
                        if client_value is not None and global_mean is not None
                        else None
                    ),
                    "difference_median": (
                        float(client_value) - global_median
                        if client_value is not None and global_median is not None
                        else None
                    ),
                    "percentile": (
                        float(
                            stats.percentileofscore(
                                non_missing_values, client_value, kind='rank'
                            )
                        )
                        if client_value is not None
                        else None
                    ),
                })

            # Si la variable est catégorielle
            elif pd.api.types.is_categorical_dtype(df_application_test[variable]) or pd.api.types.is_object_dtype(df_application_test[variable]):
                category_counts = non_missing_values.value_counts().to_dict()
                category_counts = {str(k): int(v) for k, v in category_counts.items()}

                distribution_results.append({
                    "variable": variable,
                    "type": "categorical",
                    "client_value": str(client_value) if pd.notna(client_value) else None,
                    "category_counts": category_counts,
                    "percentile": None,
                })

            else:
                raise HTTPException(status_code=400, detail=f"Type de variable non supporté : {variable}")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors du traitement de la variable {variable}: {str(e)}"
            )

    # Étape 4 : Retourner les résultats
    return {
        "client_id": request.client_id,
        "distributions": distribution_results
    }












###################################################
###################################################       FONCTION POUR DES INFORMATIONS SUR LE CREDIT       ######################################################
###################################################

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


###################################################
###################################################       FONCTION POUR DES INFORMATIONS SUR LES VARIABLES       ######################################################
###################################################


@app.get("/descriptions/")
def get_all_column_descriptions():
    """
    Retourne toutes les descriptions des colonnes disponibles.
    
    Returns:
        dict: Dictionnaire de toutes les colonnes et leurs descriptions.
    """
    descriptions = column_description["Description"].to_dict()
    return {"descriptions": descriptions}
