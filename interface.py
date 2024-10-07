import streamlit as st
import requests

#from api import get_client_data, get_credit_info, predict_credit

# Définir l'URL de base de votre API
API_URL = "http://localhost:8000"

# Fonction pour obtenir la liste des clients
def get_clients_list():
    response = requests.post(f"{API_URL}/get_clients_list")
    if response.status_code == 200:
        return response.json().get('clients_list', [])
    else:
        st.error("Impossible de récupérer la liste des clients.")
        return []

# Fonction pour obtenir les informations sur un client
def get_client_data(client_id):
    response = requests.post(f"{API_URL}/get_client_data", json={"client_id": client_id})
    if response.status_code == 200:
        return response.json().get('client_data', {})
    else:
        st.error("Impossible de récupérer les informations du client.")
        return {}

# Fonction pour obtenir les informations générales sur le crédit
def get_credit_info(client_id):
    response = requests.post(f"{API_URL}/get_credit_info", json={"client_id": client_id})
    if response.status_code == 200:
        return response.json().get('credit_info', {})
    else:
        st.error("Impossible de récupérer les informations de crédit.")
        return {}

# Fonction pour prédire le score de crédit
def predict_credit(client_id):
    response = requests.post(f"{API_URL}/predict", json={"client_id": client_id})
    if response.status_code == 200:
        result = response.json()
        return result.get('result'), result.get('proba')
    else:
        st.error("Erreur lors de la prédiction.")
        return None, None

st.title("Interface Pipeline avec Streamlit")

# Utilisation du session state pour stocker le client sélectionné
if 'selected_client' not in st.session_state:
    st.session_state.selected_client = None

# Sélection du client
clients = get_clients_list()
st.session_state.selected_client = st.selectbox("Sélectionner un client", clients, index=clients.index(st.session_state.selected_client) if st.session_state.selected_client in clients else 0)

# Afficher les informations sur le client
if st.session_state.selected_client:
    st.subheader(f"Informations sur le client {st.session_state.selected_client}")
    client_data = get_client_data(st.session_state.selected_client)
    st.write(client_data)

    # Afficher les informations de crédit
    st.subheader("Informations de crédit")
    credit_info = get_credit_info(st.session_state.selected_client)
    st.write(credit_info)

    # Prédire le score de crédit
    if st.button("Prédire le score de crédit"):
        prediction, proba = predict_credit(st.session_state.selected_client)
        if prediction is not None:
            st.write(f"Prédiction : {prediction}")
            st.write(f"Probabilité : {proba}")