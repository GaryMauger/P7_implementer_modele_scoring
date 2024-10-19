import streamlit as st
import requests
from PIL import Image

# ---------------------------------------- Configuration de l'API ----------------------------------------
API_URL = "http://localhost:8000"  # Remplacez par l'URL de votre API

# ---------------------------------------- Fonctions API ----------------------------------------
def predict_credit(client_id, seuil_nom):
    """Prédiction du score de crédit du client via l'API."""
    response = requests.post(f"{API_URL}/predict", json={"client_id": client_id, "seuil_nom": seuil_nom})
    if response.status_code == 200:
        result = response.json()
        return result.get('probability'), result.get('prediction'), result.get('features')
    else:
        st.error("Erreur lors de la prédiction.")
        return None, None, None

def get_waterfall_graph(client_id):
    """Récupération du graphique SHAP via l'API."""
    response = requests.get(f"{API_URL}/waterfall/{client_id}")
    if response.status_code == 200:
        shap_data = response.json()
        return shap_data['waterfall_image']
    else:
        st.error("Erreur lors de la récupération du graphique SHAP.")
        return None

def get_global_feature_importance_graph():
    """Récupération du graphique d'importance globale via l'API."""
    response = requests.get(f"{API_URL}/global_feature_importance")
    if response.status_code == 200:
        shap_data = response.json()
        return shap_data.get('image')  # Utilisez la clé 'image' pour accéder à l'image
    elif response.status_code == 307:
        # Gérer la redirection si nécessaire
        redirect_url = response.headers.get('Location')
        response = requests.get(redirect_url)
        if response.status_code == 200:
            shap_data = response.json()
            return shap_data.get('image')  # Utilisez la clé 'image' pour accéder à l'image
    else:
        st.error("Erreur lors de la récupération du graphique d'importance globale.")
        return None

# Fonction pour récupérer les identifiants clients depuis l'API
def fetch_client_ids():
    response = requests.get(f"{API_URL}/clients")
    if response.status_code == 200:
        return response.json().get("client_ids", [])
    else:
        st.error("Erreur lors de la récupération des identifiants clients.")
        return []

def get_client_info(client_id):
    """Récupération des informations du client via l'API."""
    response = requests.get(f"{API_URL}/client_info/{client_id}")
    if response.status_code == 200:
        client_data = response.json()
        # Récupérer la première clé dynamique
        first_key = next(iter(client_data))  # Récupérer la première clé du dictionnaire
        client_info = client_data.get(first_key, None)
        if client_info is None:
            st.error(f"Aucune information trouvée pour le client ID {client_id}.")
        return client_info
    else:
        st.error("Erreur lors de la récupération des informations client.")
        return None

def get_credit_info(client_id):
    """Récupération des informations de crédit du client via l'API."""
    response = requests.get(f"{API_URL}/credit_info/{client_id}")
    if response.status_code == 200:
        credit_data = response.json()
        # Récupérer la première clé dynamique
        first_key = next(iter(credit_data))  # Récupérer la première clé du dictionnaire
        credit_info = credit_data.get(first_key, None)
        if credit_info is None:
            st.error(f"Aucune information de crédit trouvée pour le client ID {client_id}.")
        return credit_info
    else:
        st.error("Erreur lors de la récupération des informations de crédit.")
        return None

# Fonction pour récupérer les colonnes et leurs descriptions via l'API
def get_column_descriptions():
    response = requests.get(f"{API_URL}/descriptions/")  # Changez ici
    if response.status_code == 200:
        return response.json().get('descriptions', {})
    else:
        st.error(f"Erreur lors de la récupération des descriptions. Code de statut : {response.status_code}, Contenu : {response.text}")
        return {}



############################################################################################################################################################
###################################################       INTERFACE STREAMLIT       ########################################################################
############################################################################################################################################################


# Charger le logo depuis le chemin spécifié
logo_path = r"C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Logo_pret_a_depenser.png"
logo_image = Image.open(logo_path)

# Configurer la mise en page pour qu'elle soit large
st.set_page_config(layout="wide")




# Titre de l'application
st.title("Prédiction de Scoring Crédit")

# Récupérer la liste des identifiants clients
valid_client_ids = fetch_client_ids()

# Saisie de l'identifiant client
if 'client_id' not in st.session_state:
    st.session_state.client_id = valid_client_ids[0]  # Valeur par défaut

client_id = st.selectbox("Sélectionnez l'ID du client :", valid_client_ids, index=valid_client_ids.index(st.session_state.client_id), key="client_id")

# Saisie de l'identifiant client
#client_id = st.selectbox("Sélectionnez l'ID du client :", valid_client_ids, key="client_id")

# Dictionnaire pour les seuils
seuils = {
    "Faible": "Faible risque (0.05) : On refuse le prêt dès qu'il y a 5% de probabilité que le client fasse défaut. C'est extrêmement conservateur et minimise le risque de faux négatifs, mais beaucoup de clients risquent de se voir refuser un prêt.",
    "Modéré": "Risque modéré (0.17) : On accepte un risque modéré en refusant le prêt dès que la probabilité atteint 17%. On contrôle bien les défauts, mais on peut refuser des clients peu risqués.",
    "Neutre": "Risque neutre (0.50) : Seuil standard où on classe les clients comme risqués si leur probabilité de défaut est supérieure à 50%.",
    "Elevé": "Risque élevé (0.70) : On est très tolérant au risque, et on refuse le prêt seulement si la probabilité de défaut est extrêmement élevée (70%)"
}

# Saisir le seuil avec une description
if 'seuil_nom' not in st.session_state:
    st.session_state.seuil_nom = "Faible"  # Valeur par défaut

seuil_nom = st.selectbox("Sélectionnez le seuil :", options=list(seuils.keys()), index=list(seuils.keys()).index(st.session_state.seuil_nom), key="seuil")

# Saisir le seuil avec une description
#seuil_nom = st.selectbox("Sélectionnez le seuil :", options=list(seuils.keys()), key="seuil")

# Afficher la description du seuil sélectionné
#st.write("### Description du seuil sélectionné :")
st.write(seuils[seuil_nom])

if st.button("Prédire"):
    # Appel de la fonction de prédiction
    proba, prediction, features = predict_credit(client_id, seuil_nom)

    if proba is not None:
        proba_defaut = proba * 100  # Convertir la probabilité en pourcentage
        if prediction == 0:
            # Prêt accordé (pas de défaut), affichage en vert
            st.markdown(f"<div style='background-color: green; padding: 10px; color: white; border-radius: 5px;'>"
                        f"<strong>Prédiction : Prêt accordé</strong><br>"
                        f"Probabilité de défaut : {proba_defaut:.2f}%"
                        f"</div>", unsafe_allow_html=True)
        else:
            # Prêt refusé (défaut), affichage en rouge
            st.markdown(f"<div style='background-color: red; padding: 10px; color: white; border-radius: 5px;'>"
                        f"<strong>Prédiction : Prêt refusé</strong><br>"
                        f"Probabilité de défaut : {proba_defaut:.2f}%"
                        f"</div>", unsafe_allow_html=True)

        # Afficher les features utilisées
        #st.write("**Features utilisées :**")
        #st.json(features)

        # Récupérer le graphique SHAP
        shap_image = get_waterfall_graph(client_id)
        if shap_image:
            st.image(shap_image, use_column_width=True)



# Récupérer les informations du client pour les afficher dans la sidebar
client_info = get_client_info(client_id)
credit_info = get_credit_info(client_id)
column_descriptions = get_column_descriptions()

with st.sidebar:
    st.image(logo_image, use_column_width=True)
    st.write("")
    st.markdown("---")  # Ligne de séparation    
    st.subheader("Informations sur le client")
    
    if client_info:
        # Utilisation de get() pour éviter les KeyErrors
        st.write(f"**Genre** : {client_info.get('CODE_GENDER', 'Non renseigné')}")
        st.write(f"**Nombre d'enfants** : {client_info.get('CNT_CHILDREN', 'Non renseigné')}")
        st.write(f"**Possède une voiture** : {'Oui' if client_info.get('FLAG_OWN_CAR', 'N') == 'Y' else 'Non'}")
        st.write(f"**Possède un bien immobilier** : {'Oui' if client_info.get('FLAG_OWN_REALTY', 'N') == 'Y' else 'Non'}")
        st.write(f"**État civil** : {client_info.get('NAME_FAMILY_STATUS', 'Non renseigné')}")
        st.write(f"**Type de logement** : {client_info.get('NAME_HOUSING_TYPE', 'Non renseigné')}")
        st.write(f"**Niveau d'éducation** : {client_info.get('NAME_EDUCATION_TYPE', 'Non renseigné')}")
        st.write(f"**Type de revenu** : {client_info.get('NAME_INCOME_TYPE', 'Non renseigné')}")
        st.write(f"**Occupation** : {client_info.get('OCCUPATION_TYPE', 'Non renseigné')}")
        st.write(f"**Revenu total annuel** : {client_info.get('AMT_INCOME_TOTAL', 0):,.0f} €")
    
    # Ajout d'une barre de séparation
    st.markdown("---")  # Ligne de séparation   
    st.subheader("Informations sur le crédit")
    
    if credit_info:
        st.write(f"**Type de contrat** : {credit_info.get('NAME_CONTRACT_TYPE', 'Non renseigné')}")
        st.write(f"**Montant du crédit** : {credit_info.get('AMT_CREDIT', 'Non renseigné')}")
        st.write(f"**Montant de l'annuité** : {credit_info.get('AMT_ANNUITY', 'Non renseigné')}")
        st.write(f"**Montant des biens** : {credit_info.get('AMT_GOODS_PRICE', 'Non renseigné')}")
        
    # Ajout d'une barre de séparation
    st.markdown("---")  # Ligne de séparation
    st.subheader("Descriptions des colonnes")
    
    if column_descriptions:
        # Trier les noms de colonnes par ordre alphabétique
        sorted_columns = sorted(column_descriptions.keys())
        
        # Vérifier si la colonne sélectionnée existe dans le session_state
        if 'selected_column' not in st.session_state:
            st.session_state.selected_column = sorted_columns[0]  # Valeur par défaut

        # Créer un selectbox pour choisir une colonne
        selected_column = st.selectbox("Choisissez une colonne", sorted_columns, index=sorted_columns.index(st.session_state.selected_column))
        
        # Mettre à jour la session state si la colonne sélectionnée change
        if selected_column != st.session_state.selected_column:
            st.session_state.selected_column = selected_column
            
        # Afficher la description de la colonne sélectionnée
        st.write(f"**Description :** {column_descriptions[st.session_state.selected_column]}")
    else:
        st.write("Aucune description de colonne disponible.")
    
    if column_descriptions:
        for column_name, description in sorted(column_descriptions.items()):
            st.markdown(f"**{column_name}** : {description}")  # Affiche le nom de la colonne et sa description
    else:
        st.write("Aucune description de colonne disponible.")



# Bouton pour afficher le graphique d'importance globale
if st.button("Afficher l'Importance Globale des Features"):
    st.warning("Veuillez noter que la génération de ce graphique peut prendre plusieurs minutes.")
    
    # Récupérer le graphique d'importance globale
    global_importance_graph = get_global_feature_importance_graph()

    if global_importance_graph:
        st.image(global_importance_graph, use_column_width=True, caption="Importance Globale des Features")




