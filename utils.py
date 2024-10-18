import pickle
import mlflow
import pandas as pd
import shap
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import random
import pipeline_features_eng
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------- Configuration des chemins ---------------------------------------- #

# Spécifier le chemin du modèle
model_path = "file:///C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/mlartifacts/535513794895831126/144f60891d2140538a6daad907da28a3/artifacts/model"
data_path = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/"
data_path_test = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Projet+Mise+en+prod+-+home-credit-default-risk/"

# ---------------------------------------- Chargement du modèle et des données ---------------------------------------- #

# Charger le modèle à partir du chemin local
model = mlflow.xgboost.load_model(model_path)

column_description = pd.read_csv(data_path + "Projet+Mise+en+prod+-+home-credit-default-risk/HomeCredit_columns_description.csv", 
                                     usecols=['Row', 'Description'], 
                                     index_col=0, 
                                     encoding='unicode_escape')

# ---------------------------------------- Transformation des données ---------------------------------------- #

# Les transformations appliquées aux données d'entrée
#def transform(df):
    # return transform_data(df) # L'éxécution de la fonction de transformation des données étant longue (> 10 minutes), nous chargeons directement les données transformées depuis un fichier csv.
#    return pd.read_csv(data_path + "df_data_6.csv")


# ---------------------------------------- Chargement des données clients ---------------------------------------- #

# Chargement des données des clients depuis un fichier CSV
df_clients = pipeline_features_eng.execute_pipeline()
df_application_test = pd.read_csv(data_path_test + 'application_test.csv')

# Filtrer les clients selon SK_ID_CURR
clients_data = df_clients[df_clients['SK_ID_CURR'].isin(df_application_test['SK_ID_CURR'])]

# Créer une instance de MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Colonnes à exclure de la mise à l'échelle
exclude_cols = ['SK_ID_CURR', 'TARGET']

# Sélectionner les colonnes numériques à scaler tout en excluant les colonnes spécifiées
columns_to_scale = clients_data.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)

# Appliquer le scaler uniquement sur les colonnes sélectionnées
clients_data_scaled = clients_data.copy()  # Crée une copie pour garder df_clients propre
clients_data_scaled[columns_to_scale] = scaler.fit_transform(clients_data[columns_to_scale])


#clients_data.info()

#feats = [f for f in clients_data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index','Unnamed: 0']]
feats = [f for f in clients_data.columns if f not in ['TARGET','SK_ID_CURR']]


# ---------------------------------------- Fonctions de traitements des données ---------------------------------------- #

def get_all_features():
    """Récupère la liste de toutes les colonnes sans doublons et triées par ordre alphabétique."""
    all_features = column_description.index.tolist()  # Supposons que vous ayez un index de DataFrame
    unique_features = list(set(all_features))  # Supprime les doublons
    unique_features.sort()  # Trie par ordre alphabétique
    return unique_features

def description(column_name):
    """
    Retourne la description d'une colonne spécifique depuis le fichier CSV.
    
    Args:
    column_name (str): Le nom de la colonne pour laquelle obtenir la description.
    
    Returns:
    str: La description de la colonne.
    """
    try:
        # Vérifier si le nom de colonne demandé existe dans l'index
        if column_name not in column_description.index:
            return f"Aucune description trouvée pour la colonne '{column_name}'."
        
        # Retourner la description depuis l'index
        return column_description.loc[column_name, 'Description']
    
    except Exception as e:
        return f"Erreur lors de la récupération de la description : {e}"



def get_all_clients():
    """
    Retourne la liste de tous les ID clients à partir des données clients.
    """
    # Assurez-vous que clients_data est chargé ici, par exemple à partir d'un fichier CSV ou d'une base de données
    # clients_data = pd.read_csv('path_to_your_client_data.csv')
    
    # On suppose que clients_data est déjà disponible sous forme de DataFrame
    return clients_data['SK_ID_CURR'].tolist()

#def clients_id_list():
    """
    Retourne une liste de 10 ID clients aléatoires à partir des données clients filtrées.
    """
#    return random.sample(clients_data['SK_ID_CURR'].tolist(), 10)

def credit_info(client_id):
    """
    Retourne les informations de crédit pour le client sélectionné.
    """
    credit_info_columns = ["NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    credit_info = clients_data.loc[clients_data['SK_ID_CURR'] == client_id, credit_info_columns].T
    return credit_info

def client_info(client_id):
    """
    Retourne les informations personnelles du client sélectionné.
    """
    client_info_columns = [
        "CODE_GENDER", "CNT_CHILDREN", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "AMT_INCOME_TOTAL"
    ]
    client_info = df_application_test.loc[df_application_test['SK_ID_CURR'] == client_id, client_info_columns].T
    client_info = client_info.fillna('N/A')  # Remplacer les valeurs manquantes par 'N/A'
    return client_info


# ---------------------------------------- Fonctions de prédiction ---------------------------------------- #

# Définition des seuils avec leurs noms associés du business score
thresholds = {
    "sans": {"valeur": 0.05, "nom": "Sans risque"},
    "faible": {"valeur": 0.17, "nom": "Faible coût"},
    "modere": {"valeur": 0.50, "nom": "Coût modéré"},
    "eleve": {"valeur": 0.28, "nom": "Coût élevé"}
}


def predict(client_id, seuil_nom="faible"):
    """
    Effectue la prédiction du score de crédit pour un client donné en utilisant le seuil spécifié.
    Retourne la probabilité, la classe prédite (0 ou 1), les features utilisées et le seuil.
    """
    # Récupérer le client sélectionné dans le DataFrame
    selected_client = clients_data_scaled.loc[clients_data_scaled['SK_ID_CURR'] == client_id]
    
    # Assurez-vous que 'selected_client' contient bien un client
    if selected_client.empty:
        raise ValueError(f"Aucun client trouvé avec SK_ID_CURR = {client_id}")
    
    # Calcul de la probabilité de la classe positive
    proba = model.predict_proba(selected_client[feats])[0][1]  # Probabilité de défaut
    
    # Récupérer le seuil correspondant et son nom
    seuil_info = thresholds[seuil_nom]
    seuil_valeur = seuil_info["valeur"]
    seuil_nom_affiche = seuil_info["nom"]
    
    # Faire la prédiction en fonction du seuil
    prediction = int(proba >= seuil_valeur)  # Prédiction basée sur le seuil choisi (0 ou 1)
    
    # Retourner la probabilité, la prédiction, les features utilisées, et le seuil avec son nom
    return proba, prediction, selected_client[feats], seuil_valeur, seuil_nom_affiche


def encode_image_to_base64(fig):
    """Encode une figure Matplotlib en base64."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return f"data:image/png;base64,{image_base64}"

#import matplotlib
#matplotlib.use('Agg')

def shap_waterfall_chart_bis(selected_client, model, feat_number=10):
    """
    Génère un graphique waterfall SHAP pour un client spécifique.
    """
    # Vérifier que les données du client ne contiennent pas de colonne TARGET ou SK_ID_CURR
    selected_client = selected_client.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    
    # Utiliser SHAP TreeExplainer pour le modèle
    explainer = shap.TreeExplainer(model)
    
    # Calculer les valeurs SHAP pour les données du client
    shap_values = explainer.shap_values(selected_client)
    
    # Pour les modèles de classification binaire, shap_values est une liste
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Classe positive
    
    plt.ioff()
    
    # Générer le graphique waterfall SHAP
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                          base_values=explainer.expected_value,
                                          data=selected_client.iloc[0], 
                                          feature_names=selected_client.columns),
                        max_display=feat_number)
    
    # Convertir le graphique en image encodée en base64
    image_base64 = encode_image_to_base64(plt)
    
    # Fermer la figure pour éviter d'accumuler des figures ouvertes
    plt.close()
    
    return image_base64

from streamlit_shap import st_shap

def shap_waterfall_chart(selected_client, model, feat_number=10):
    """
    Génère un graphique waterfall SHAP pour un client spécifique et l'affiche directement.
    """
    # Vérifier que les données du client ne contiennent pas de colonne TARGET ou SK_ID_CURR
    selected_client = selected_client.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    
    # Utiliser SHAP TreeExplainer pour le modèle
    explainer = shap.TreeExplainer(model)
    
    # Calculer les valeurs SHAP pour les données du client
    shap_values = explainer.shap_values(selected_client)
    
    # Pour les modèles de classification binaire, shap_values est une liste
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Classe positive

    # Créer un graphique waterfall SHAP
    shap_explanation = shap.Explanation(values=shap_values[0],
                                         base_values=explainer.expected_value,
                                         data=selected_client.iloc[0], 
                                         feature_names=selected_client.columns)

    # Utiliser st_shap pour afficher le graphique dans Streamlit
    st_shap(shap.waterfall_plot(shap_explanation, max_display=feat_number))


def calculate_shap_values(selected_client, model):
    """
    Calcule et retourne les valeurs SHAP pour le client sélectionné, ainsi que les valeurs de base.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(selected_client)

    # Récupérer les valeurs de base
    base_values = explainer.expected_value  # Cela renvoie les valeurs de base

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Classe positive

    # Conversion en types natifs
    if isinstance(shap_values, np.ndarray):
        shap_values = shap_values.tolist()  # Convertir les valeurs SHAP en liste
    if isinstance(base_values, np.ndarray):
        base_values = float(base_values)  # Convertir en float si c'est un tableau

    return shap_values, base_values  # Retourner les valeurs SHAP et de base
    
