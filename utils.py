import pickle
import mlflow
import pandas as pd
import random
import pipeline_features_eng


# ---------------------------------------- Configuration des chemins ---------------------------------------- #

# Spécifier le chemin du modèle
model_path = "file:///C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/mlartifacts/950890628191069861/e31f1e6f9ec0467f87578900af3c4d68/artifacts/model"
data_path = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/"

# ---------------------------------------- Chargement du modèle et des données ---------------------------------------- #

# Charger le modèle à partir du chemin local
model = mlflow.xgboost.load_model(model_path)

column_description = pd.read_csv(data_path + "Projet+Mise+en+prod+-+home-credit-default-risk/HomeCredit_columns_description.csv", 
                                     usecols=['Row', 'Description'], 
                                     index_col=0, 
                                     encoding='unicode_escape')

# ---------------------------------------- Transformation des données ---------------------------------------- #

# Les transformations appliquées aux données d'entrée
def transform(df):
    # return transform_data(df) # L'éxécution de la fonction de transformation des données étant longue (> 10 minutes), nous chargeons directement les données transformées depuis un fichier csv.
    return pd.read_csv(data_path + "df_data_6.csv")


# ---------------------------------------- Chargement des données clients ---------------------------------------- #

# Chargement des données des clients depuis un fichier CSV
#prod_data = pd.read_csv(data_path + "application_test.csv") # base de clients en "production", nouveaux clients
df_clients = pipeline_features_eng.execute_pipeline()
#df_data_6 = transform(prod_data) # Transformation des données clients pour utilisation du modèle

df_application_test = pd.read_csv(data_path + 'application_test.csv')

clients_data = df_clients[df_clients['SK_ID_CURR'].isin(df_application_test['SK_ID_CURR'])]

#clients_data.info()

feats = [f for f in clients_data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index','Unnamed: 0']]


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

def predict(client_id):
    """
    Effectue la prédiction du score de crédit pour un client donné.
    Retourne la probabilité et la classe prédite (0 ou 1).
    """
    selected_client = clients_data.loc[clients_data['SK_ID_CURR'] == client_id]
    proba = model.predict_proba(selected_client[feats]).tolist()[0][0]
    prediction = model.predict(selected_client[feats]).tolist()[0]
    return proba, prediction
