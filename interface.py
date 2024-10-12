# Pour démarrer l'application Streamlit, exécuter la commande suivante dans le terminal : streamlit run interface.py


import streamlit as st
import requests
from PIL import Image

# URL de base de l'API
API_URL = "http://localhost:8000"

# Configurer la mise en page pour qu'elle soit large
st.set_page_config(layout="wide")

# ---------------------------------------- Fonctions API ---------------------------------------- #

def get_clients_list():
    """Récupère la liste des clients depuis l'API."""
    response = requests.post(f"{API_URL}/get_clients_list")
    if response.status_code == 200:
        return response.json().get('clients_list', [])
    else:
        st.error("Impossible de récupérer la liste des clients.")
        return []

def get_client_data(client_id):
    """Récupère les informations du client sélectionné depuis l'API."""
    response = requests.post(f"{API_URL}/get_client_data", json={"client_id": client_id})
    if response.status_code == 200:
        return response.json().get('client_data', {})
    else:
        st.error("Impossible de récupérer les informations du client.")
        return {}

def get_credit_info(client_id):
    """Récupère les informations de crédit du client depuis l'API."""
    response = requests.post(f"{API_URL}/get_credit_info", json={"client_id": client_id})
    if response.status_code == 200:
        return response.json().get('credit_info', {})
    else:
        st.error("Impossible de récupérer les informations de crédit.")
        return {}

def predict_credit(client_id):
    """Prédiction du score de crédit du client via l'API."""
    response = requests.post(f"{API_URL}/predict", json={"client_id": client_id})
    if response.status_code == 200:
        result = response.json()
        return result.get('result'), result.get('proba')
    else:
        st.error("Erreur lors de la prédiction.")
        return None, None

def get_column_description(column_name):
    """Récupère la description d'une colonne depuis l'API."""
    response = requests.get(f"{API_URL}/get_column_description/{column_name}")
    if response.status_code != 200:
        #st.error("Erreur lors de la récupération de la description de la colonne.")
        return "Pas de définition disponible."

    json_response = response.json()

    # Vérifiez si la clé 'column_description' est présente
    if "column_description" in json_response:
        return json_response["column_description"]
    else:
        st.warning(f"Aucune définition trouvée pour la colonne '{column_name}'.")
        return "Pas de définition disponible."


def get_all_features():
    """Récupère la liste de toutes les colonnes depuis l'API."""
    response = requests.get(f"{API_URL}/get_all_features")  # Assurez-vous que l'endpoint est correct
    if response.status_code == 200:
        return response.json().get("columns", [])
    else:
        st.error("Impossible de récupérer la liste des colonnes.")
        return []


# ---------------------------------------- Interface Streamlit ---------------------------------------- #

# Charger le logo depuis le chemin spécifié
logo_path = r"C:/Users/mauge\Documents/github/P7_implementer_modele_scoring/Logo_pret_a_depenser.png"
logo_image = Image.open(logo_path)

# Afficher le logo en haut de la page
#st.image(logo_image, width=900)

#st.title("Aide à la décision pour l'octroi d'un crédit")


# 1. Section : Saisie du numéro du client
clients = get_clients_list()  # Obtenir la liste complète des clients

# Créer une ligne de colonnes pour l'entrée du numéro de client et les informations sur le client
col1, col2 = st.columns([2, 2])  # Ajustez les proportions si nécessaire

with col1:
    client_id_input = st.text_input("Entrez le numéro du client souhaité :")
    
    # Afficher le logo en haut de la page
    st.image(logo_image, width=900)

    if st.button("Valider"):
        if client_id_input:
            client_id_input = int(client_id_input)  # Convertir l'input en entier
            # Vérifier si le client existe dans la liste complète
            if client_id_input in clients:
                st.session_state.selected_client = client_id_input

                # Charger les données du client et les informations de crédit
                st.session_state.client_data = get_client_data(st.session_state.selected_client)
                st.session_state.credit_info = get_credit_info(st.session_state.selected_client)

                # Afficher les informations sur le client
                if st.session_state.client_data:
                    st.subheader(f"Informations sur le client")
                    with st.expander("Détails du client"):
                        st.write(st.session_state.client_data)

                # Afficher les informations de crédit
                if st.session_state.credit_info:
                    st.subheader("Informations de crédit")
                    with st.expander("Détails du crédit"):
                        st.write(st.session_state.credit_info)

                # Prédiction du score de crédit
                st.subheader("Prédiction du score de crédit")
                prediction, proba = predict_credit(st.session_state.selected_client)
                if prediction is not None:
                    proba_defaut = (1 - proba) * 100  # Supposant que proba est la probabilité de non-défaut
                    st.write(f"Prédiction : **{'Prêt accordé' if prediction == 0 else 'Prêt refusé'}**")
                    st.write(f"Probabilité de défaut : **{proba_defaut:.2f}%**")
            else:
                st.error("Le numéro de client saisi n'est pas dans la liste des clients.")
        else:
            st.error("Veuillez entrer un numéro de client.")

with col2:
    
    st.title("Aide à la décision pour l'octroi d'un crédit")
    
    # Afficher les informations sur le client si déjà sélectionné
    if "selected_client" in st.session_state:
        # Afficher les informations sur le client
        if st.session_state.client_data:
            st.subheader(f"Informations sur le client")
            with st.expander("Détails du client"):
                st.write(st.session_state.client_data)

        # Afficher les informations de crédit
        if st.session_state.credit_info:
            st.subheader("Informations de crédit")
            with st.expander("Détails du crédit"):
                st.write(st.session_state.credit_info)



# ---------------------------------------- Description des colonnes ---------------------------------------- #

st.subheader("Liste des colonnes disponibles")

# Récupérer la liste de toutes les colonnes
columns = get_all_features()  # Appeler la nouvelle fonction

# Vérifiez si la liste des colonnes n'est pas vide
if columns:
    # Afficher une liste déroulante pour sélectionner une colonne
    selected_column = st.selectbox("Sélectionnez une colonne :", columns)

    # Afficher la description de la colonne sélectionnée
    if selected_column:
        description = get_column_description(selected_column)
        if description is not None:
            st.write(f"**Description de la colonne '{selected_column}':** {description}")
else:
    st.write("Aucune colonne disponible.")
