import pickle
import mlflow
import pandas as pd
import random
import pipeline_features_eng

# Spécifier le chemin du modèle
model_path = "file:///C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/mlartifacts/950890628191069861/e31f1e6f9ec0467f87578900af3c4d68/artifacts/model"
data_path = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/"

# Charger le modèle à partir du chemin local
model = mlflow.xgboost.load_model(model_path)

# Les transformations appliquées aux données d'entrée
def transform(df):
    # return transform_data(df) # L'éxécution de la fonction de transformation des données étant longue (> 10 minutes), nous chargeons directement les données transformées depuis un fichier csv.
    return pd.read_csv(data_path + "df_data_6.csv")

# Chargement des données des clients depuis un fichier CSV
#prod_data = pd.read_csv(data_path + "application_test.csv") # base de clients en "production", nouveaux clients
df_clients = pipeline_features_eng.execute_pipeline()
#df_data_6 = transform(prod_data) # Transformation des données clients pour utilisation du modèle

df_application_test = pd.read_csv(data_path + 'application_test.csv')

clients_data = df_clients[df_clients['SK_ID_CURR'].isin(df_application_test['SK_ID_CURR'])]

clients_data.info()


feats = [f for f in clients_data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index','Unnamed: 0']]

# Fonction de prédiction
def predict(client_id):
    # Simulation de la récupération des données client et prédiction
    selected_client = clients_data.loc[clients_data['SK_ID_CURR']==client_id]
    proba = model.predict_proba(selected_client[feats]).tolist()[0][0]
    prediction = model.predict(selected_client[feats]).tolist()[0]
    return proba, prediction

# Fonction pour obtenir une liste de 10 ID clients aléatoires
def clients_id_list():
    return random.sample(clients_data['SK_ID_CURR'].tolist(), 10)

# Retourne les caractéristiques du crédit demandé par le client
def credit_info(client_id):
    credit_info_columns=["NAME_CONTRACT_TYPE","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]
    credit_info=clients_data.loc[clients_data['SK_ID_CURR']==client_id,credit_info_columns].T # informations crédit pour le client selectionné
    return credit_info

# Retourne les informations personnelles sur le client
def client_info(client_id):
    client_info_columns = [
                 "CODE_GENDER",
                 "CNT_CHILDREN",
                 "FLAG_OWN_CAR",
                 "FLAG_OWN_REALTY",
                 "NAME_FAMILY_STATUS",
                 "NAME_HOUSING_TYPE",
                 "NAME_EDUCATION_TYPE",
                 "NAME_INCOME_TYPE",
                 "OCCUPATION_TYPE",
                 "AMT_INCOME_TOTAL"
                 ]    
    client_info=df_application_test.loc[df_application_test['SK_ID_CURR']==client_id,client_info_columns].T # informations client pour le client selectionné
    client_info= client_info.fillna('N/A')
    return client_info


# Exemple d'utilisation des fonctions

# Charger le modèle et les données
print("Liste de 10 clients aléatoires :", clients_id_list())

# Choisir un client au hasard parmi la liste
client_id = clients_id_list()[0]
print("Client sélectionné :", client_id)

# Tester la prédiction pour ce client
proba, prediction = predict(client_id)
print("Prédiction pour le client :", prediction)
print("Probabilité associée :", proba)

# Récupérer les informations crédit et client
credit_info_client = credit_info(client_id)
print("Informations crédit pour le client :\n", credit_info_client)

client_info_client = client_info(client_id)
print("Informations personnelles pour le client :\n", client_info_client)
