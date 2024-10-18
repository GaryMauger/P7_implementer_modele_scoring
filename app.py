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
client_id = st.selectbox("Sélectionnez l'ID du client :", valid_client_ids)

# Saisie de l'identifiant client
#client_id = st.number_input("Entrez l'ID du client :", min_value=0)

# Saisir le seuil
seuil_nom = st.selectbox("Sélectionnez le seuil :", options=["sans", "faible", "modere", "eleve"])

if st.button("Prédire"):
    # Appel de la fonction de prédiction
    proba, prediction, features = predict_credit(client_id, seuil_nom)

    if proba is not None:
        # Affichage des résultats de la prédiction
        st.write(f"**Probabilité de défaut :** {proba:.2f}")
        st.write(f"**Prédiction :** {'Défaut' if prediction == 1 else 'Pas de défaut'}")
        
        # Afficher les features utilisées
        st.write("**Features utilisées :**")
        st.json(features)

        # Récupérer le graphique SHAP
        shap_image = get_waterfall_graph(client_id)
        if shap_image:
            st.image(shap_image, use_column_width=True)


with st.sidebar:
    st.image(logo_image, use_column_width=True)
    
    st.subheader("Informations sur le client et le crédit")
