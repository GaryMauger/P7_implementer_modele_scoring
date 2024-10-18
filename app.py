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




# ---------------------------------------- Interface Streamlit ----------------------------------------
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
client_id = st.selectbox("Sélectionnez l'ID du client :", valid_client_ids, key="client_id")

# Dictionnaire pour les seuils
#seuils = {
#    "sans": "Seuil très bas (risque élevé de faux positifs)",
#    "faible": "Minimiser le Coût tout en Conservant une Bonne Précision",
#    "modere": "Tolérer un Coût Modéré pour Améliorer la Précision",
#    "eleve": "Réduire le Risque des Faux Négatifs à Tout Prix"
#}

# Saisir le seuil
#seuil_nom = st.selectbox("Sélectionnez le seuil :", options=list(seuils.keys()), format_func=lambda x: seuils[x], key="seuil")

# Dictionnaire pour les seuils
seuils = {
    "Sans": "Sans seuil : Classe presque tous les clients comme à risque, entraînant un nombre élevé de faux positifs et des coûts importants. À éviter.",
    "Faible": "Faible : Bon compromis entre coût et précision (Coût : 30 295 €, Précision : 86 %). Idéal pour équilibrer détection des défauts et rentabilité.",
    "Modéré": "Modéré : Maximiser la précision (Coût : 36 725 €, Précision : 92 %), mais augmente les faux négatifs. Convient si la réputation est primordiale.",
    "Elevé": "Élevé : Réduit le risque de faux négatifs (Coût : 101 570 €, Précision : 91 %), adapté pour éviter les pertes financières majeures."
}

# Saisir le seuil avec une description
seuil_nom = st.selectbox("Sélectionnez le seuil :", options=list(seuils.keys()), key="seuil")

# Afficher la description du seuil sélectionné
#st.write("### Description du seuil sélectionné :")
st.write(seuils[seuil_nom])

if st.button("Prédire"):
    # Appel de la fonction de prédiction
    proba, prediction, features = predict_credit(client_id, seuil_nom)

    if proba is not None:
        # Affichage des résultats de la prédiction
        st.write(f"**Probabilité de défaut :** {proba:.2f}")
        st.write(f"**Prédiction :** {'Défaut' if prediction == 1 else 'Pas de défaut'}")
        
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

