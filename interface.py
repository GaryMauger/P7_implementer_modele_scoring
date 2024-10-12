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
st.markdown(
    "<h1 style='text-align: center; font-size: 3em;'>Aide à la décision pour l'octroi d'un crédit</h1>",
    unsafe_allow_html=True
)


# 1. Section : Saisie du numéro du client
# Vérifier si la liste des clients est déjà chargée dans la session
if "clients" not in st.session_state:
    st.session_state.clients = get_clients_list()  # Obtenir la liste complète des clients

# Créer une ligne de colonnes pour l'entrée du numéro de client et les informations sur le client
col1, col2 = st.columns([2, 2])  # Ajustez les proportions si nécessaire

with col1:
    st.image(logo_image, width=1000)
    st.write(" ")  # ligne vide pour laisser un espace

    client_id_input = st.text_input("Entrez le numéro du client souhaité :")

    if st.button("Valider"):
        if client_id_input:
            client_id_input = int(client_id_input)  # Convertir l'input en entier
            # Vérifier si le client existe dans la liste complète
            if client_id_input in st.session_state.clients:
                st.session_state.selected_client = client_id_input

                # Charger les données du client et les informations de crédit
                if "client_data" not in st.session_state or st.session_state.selected_client != client_id_input:
                    st.session_state.client_data = get_client_data(st.session_state.selected_client)
                    st.session_state.credit_info = get_credit_info(st.session_state.selected_client)

                # Prédiction du score de crédit
                if "prediction" not in st.session_state or st.session_state.selected_client != client_id_input:
                    st.session_state.prediction, st.session_state.proba = predict_credit(st.session_state.selected_client)

                # Afficher la prédiction si disponible
                if st.session_state.prediction is not None:
                    # Convertir la probabilité en pourcentage pour le défaut
                    proba_defaut = (1 - st.session_state.proba) * 100  # Supposant que proba est la probabilité de non-défaut
                    if st.session_state.prediction == 0:
                        # Prêt accordé
                        st.markdown(
                            f"<div style='background-color: green; padding: 10px; color: white; border-radius: 5px;'>"
                            f"<strong>Prédiction : Prêt accordé</strong><br>"
                            f"Probabilité de défaut : {proba_defaut:.2f}%"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        # Prêt refusé
                        st.markdown(
                            f"<div style='background-color: red; padding: 10px; color: white; border-radius: 5px;'>"
                            f"<strong>Prédiction : Prêt refusé</strong><br>"
                            f"Probabilité de défaut : {proba_defaut:.2f}%"
                            f"</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.error("Le numéro de client saisi n'est pas dans la liste des clients.")
        else:
            st.error("Veuillez entrer un numéro de client.")

with col2:
    st.write(" ")  # ligne vide pour laisser un espace

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

# Utiliser la sidebar pour afficher la liste des colonnes disponibles
with st.sidebar:
    st.subheader("Liste des colonnes disponibles")

    # Vérifier si la liste des colonnes est déjà chargée dans la session
    if "columns" not in st.session_state:
        st.session_state.columns = get_all_features()  # Charger les colonnes une seule fois

    # Vérifiez si la liste des colonnes n'est pas vide
    if st.session_state.columns:
        # Afficher une liste déroulante pour sélectionner une colonne
        selected_column = st.selectbox("Sélectionnez une colonne :", st.session_state.columns)

        # Stocker la description de la colonne sélectionnée dans la session
        if "selected_column_description" not in st.session_state:
            st.session_state.selected_column_description = ""

        if selected_column:
            # Vérifier si la description est déjà stockée
            if selected_column != st.session_state.selected_column_description:
                st.session_state.selected_column_description = get_column_description(selected_column)

            # Afficher la description de la colonne sélectionnée
            description = st.session_state.selected_column_description
            if description is not None:
                st.write(f"**Description de la colonne '{selected_column}':** {description}")
    else:
        st.write("Aucune colonne disponible.")
