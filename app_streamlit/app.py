import streamlit as st
import requests
from PIL import Image
import pandas as pd  # Bibliothèque pour manipuler des DataFrames
import plotly.express as px  # Bibliothèque pour des graphiques interactifs
import plotly.graph_objects as go

# ---------------------------------------- Configuration de l'API ----------------------------------------
API_URL = "http://localhost:8000"  # Remplacez par l'URL de votre API
#API_URL = "https://fastapi-credit-scoring-c3h7f2hfd3behne7.westeurope-01.azurewebsites.net"

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


def get_top_features(client_id):
    """Récupération des 10 variables les plus influentes via l'API."""
    response = requests.get(f"{API_URL}/top_features/{client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de la récupération des top features. (Code: {response.status_code})")
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
    

def get_client_position(client_id, variables):
    """
    Récupération de la position d'un client par rapport à la moyenne et la médiane pour les variables spécifiées via l'API.
    """
    response = requests.post(
        f"{API_URL}/client_position",
        json={"client_id": client_id, "variables": variables}
    )

    if response.status_code == 200:
        # Convertir la réponse JSON
        position_data = response.json()
        if "comparisons" not in position_data:
            st.error(f"Aucune donnée de comparaison disponible pour le client ID {client_id}.")
            return None
        return position_data  # Retourne toutes les données de la réponse
    else:
        # Gérer les erreurs de requête
        st.error(f"Erreur lors de la récupération des comparaisons. (Code: {response.status_code})")
        return None


@st.cache_data
def fetch_available_variables(client_id):
    """
    Récupère dynamiquement les variables disponibles pour un client spécifique via l'API.
    Retourne les noms des variables et leurs types (numérique ou catégoriel).
    """
    response = requests.get(f"{API_URL}/available_variables/{client_id}")
    if response.status_code == 200:
        # Extraire les variables avec leur type depuis la réponse JSON
        variables_data = response.json().get("variables", [])
        return variables_data  # Retourne la liste complète des variables avec leur type
    else:
        st.error(f"Erreur lors de la récupération des variables disponibles. (Code: {response.status_code})")
        return []



@st.cache_data(show_spinner=False)
def get_client_distribution(client_id, variables):
    """
    Récupération de la distribution d'un client par rapport aux variables spécifiées via l'API.
    """
    try:
        response = requests.post(
            f"{API_URL}/client_distribution",
            json={"client_id": client_id, "variables": variables}
        )

        if response.status_code == 200:
            # Convertir la réponse JSON
            distribution_data = response.json()

            # Vérifier que les données de distribution sont présentes
            if "distributions" not in distribution_data:
                st.error(f"Aucune donnée de distribution disponible pour le client ID {client_id}.")
                return None

            # Retourner les données
            return distribution_data

        elif response.status_code == 404:
            st.error(f"Client ID {client_id} introuvable dans la base de données.")
            return None

        elif response.status_code == 400:
            st.error("Certaines variables demandées sont invalides ou manquantes.")
            return None

        else:
            # Gérer les erreurs non spécifiées
            st.error(f"Erreur inconnue : {response.status_code}. Message : {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {str(e)}")
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

# Fonction pour afficher une jauge sous forme de barre horizontale
def afficher_jauge_horizontale(score_client, seuil_score, prediction):
    # Couleur de la jauge en fonction de la prédiction
    couleur = "green" if prediction == 0 else "red"
    
    # Création de la jauge sous forme de barre
    fig = go.Figure(go.Indicator(
        mode="gauge+number",  # Mode avec jauge et valeur numérique
        value=score_client,  # Score client
        #number={'suffix': "/100"},  # Ajouter le suffixe "/100"
        gauge={
            'axis': {'range': [25, 100]},  # Plage de la jauge
            'bar': {'color': couleur, 'thickness':1},  # Couleur de la barre de jauge
            'threshold': {
                'line': {'color': "black", 'width': 7},  # Ligne verticale pour le seuil
                'thickness': 1,  # Épaisseur de la barre
                'value': seuil_score  # Position du seuil
            }
        }
    ))
    
    # Orientation horizontale et mise en page
    fig.update_layout(
        width=1200,  # Largeur de la jauge
        height=230,  # Hauteur de la jauge
        margin=dict(l=20, r=20, t=25, b=25),  # Marges
    )
    return fig


# Initialisation ou récupération de l'état global
def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "data_received": False,
            "data": None,
            "last_client_id": None,
            "last_seuil_nom": "Modéré",  # Suivre le dernier seuil utilisé
        }
    return st.session_state["state"]

state = get_state()





############################################################################################################################################################
###################################################       INTERFACE STREAMLIT       ########################################################################
############################################################################################################################################################


# Charger le logo depuis le chemin spécifié
#logo_path = r"C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Logo_pret_a_depenser.png"
#logo_path = "./Data/"
logo_path = "./Logo_pret_a_depenser.png"
logo_image = Image.open(logo_path)

# Configurer la mise en page pour qu'elle soit large
st.set_page_config(layout="wide")




# Titre de l'application
st.title("Prédiction de Scoring Crédit")

# Récupérer la liste des identifiants clients
valid_client_ids = fetch_client_ids()

# Sélection de l'ID client avec une validation basée sur l'état
client_id = st.selectbox(
    "Sélectionnez l'ID du client :", 
    valid_client_ids, 
    index=valid_client_ids.index(state["last_client_id"]) if state["last_client_id"] in valid_client_ids else 0,
)
# Mise à jour de l'état si l'ID client change
if client_id != state["last_client_id"]:
    state["last_client_id"] = client_id
    state["data_received"] = False  # Réinitialiser l'état de réception des données


# Saisie de l'identifiant client
#client_id = st.selectbox("Sélectionnez l'ID du client :", valid_client_ids, key="client_id")

# Dictionnaire pour les seuils
seuils = {
    "Faible": "Faible risque (0.05) : On refuse le prêt dès qu'il y a 5% de probabilité que le client fasse défaut. C'est extrêmement conservateur et minimise le risque de faux négatifs, mais beaucoup de clients risquent de se voir refuser un prêt.",
    "Modéré": "Risque modéré (0.17) : On accepte un risque modéré en refusant le prêt dès que la probabilité atteint 17%. On contrôle bien les défauts, mais on peut refuser des clients peu risqués.",
    "Neutre": "Risque neutre (0.50) : Seuil standard où on classe les clients comme risqués si leur probabilité de défaut est supérieure à 50%.",
    "Elevé": "Risque élevé (0.70) : On est très tolérant au risque, et on refuse le prêt seulement si la probabilité de défaut est extrêmement élevée (70%)"
}

# Définir les seuils
thresholds = {
    "Faible": {"valeur": 0.05, "nom": "Très faible risque de refus"},
    "Modéré": {"valeur": 0.17, "nom": "Risque modéré de refus"},
    "Neutre": {"valeur": 0.50, "nom": "Risque neutre de refus"},
    "Elevé": {"valeur": 0.70, "nom": "Risque élevé de refus"}
}

# Sélection du seuil
seuil_nom = st.selectbox(
    "Sélectionnez le seuil :", 
    options=list(thresholds.keys()), 
    index=list(thresholds.keys()).index(state.get("last_seuil_nom", "Modéré")),
)

# Si le seuil change, réinitialiser les données
if seuil_nom != state.get("last_seuil_nom"):
    state["last_seuil_nom"] = seuil_nom
    state["data_received"] = False  # Forcer un nouvel appel API




# Afficher la description du seuil sélectionné
st.write(f"*{seuils[seuil_nom]}*")


 

# CSS pour personnaliser le bouton
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #6c757d;
        color: white;
        font-size: 100px;
        padding: 20px 48px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background-color: #5a6268;
    }
    </style>
""", unsafe_allow_html=True)

# Bouton "Prédire" et gestion de l'état
if st.button("Prédire") or state["data_received"]:
    # Vérifier si les données doivent être recalculées
    if not state["data_received"]:
        response = requests.post(
            f"{API_URL}/predict", 
            json={"client_id": client_id, "seuil_nom": seuil_nom}
        )
        if response.status_code == 200:
            state["data"] = response.json()
            state["data_received"] = True
        else:
            st.error(f"Erreur lors de l'appel API : {response.status_code}")
            state["data"] = None
            state["data_received"] = False

    # Utiliser les données pour afficher les résultats
    if state["data"]:
        proba = state["data"].get("probability")
        prediction = state["data"].get("prediction")
        features = state["data"].get("features")

        if proba is not None:
            proba_defaut = proba * 100  # Convertir la probabilité en pourcentage
            score_client = 100 - proba_defaut  # Calculer le score inversé
            seuil_valeur = thresholds[seuil_nom]["valeur"] * 100
            seuil_score = 100 - seuil_valeur
            distance_seuil = abs(proba_defaut - seuil_valeur)

            # Affichage des résultats
            col1, col2 = st.columns([2, 1])
            with col1:
                if prediction == 0:
                    st.success(
                        f"**Prêt accordé 🎉**\n\n"
                        f"Probabilité de défaut : **{proba_defaut:.2f}%**\n\n"
                        f"Score client : **{score_client:.2f}/100**\n\n"
                        f"Distance par rapport au seuil choisi ({seuil_score}) : **{distance_seuil:.2f}** points\n\n"
                        f"*Le risque de défaut est faible et en dessous du seuil choisi. Ce client est éligible au crédit.*"
                    )
                else:
                    st.error(
                        f"**Prêt refusé ❌**\n\n"
                        f"Probabilité de défaut : **{proba_defaut:.2f}%**\n\n"
                        f"Score client : **{score_client:.2f}/100**\n\n"
                        f"Distance par rapport au seuil choisi ({seuil_score}) : **{distance_seuil:.2f}** points\n\n"
                        f"*Le risque de défaut est élevé et dépasse le seuil choisi. Ce client n'est pas éligible au crédit.*"
                    )
            with col2:
                fig = afficher_jauge_horizontale(score_client, seuil_score, prediction)
                st.plotly_chart(fig, use_container_width=True)


        st.markdown("---")  # Ligne de séparation



       # Afficher les features influentes
        st.subheader("Interprétation du Score")
        st.write("Les principales variables influençant cette décision sont illustrées ci-dessous.")


        # Récupérer les top features
        top_features = get_top_features(client_id)


        if top_features:
            # Convertir les données des top features en DataFrame
            top_positive_df = pd.DataFrame(top_features["top_positive"])
            top_negative_df = pd.DataFrame(top_features["top_negative"])

            # Trier par SHAP Value pour garantir un ordre correct
            top_positive_df = top_positive_df.sort_values(by="SHAP Value", ascending=True)
            top_negative_df = top_negative_df.sort_values(by="SHAP Value", ascending=False)

            # Graphiques interactifs
            fig_positive = px.bar(
                top_positive_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                title="Top 10 Variables Positives",
                labels={"SHAP Value": "", "Feature": ""},
                color="SHAP Value",
                color_continuous_scale="Blues"
            )
            # Supprimer la légende et la barre de couleur
            fig_positive.update_coloraxes(showscale=False)

            fig_negative = px.bar(
                top_negative_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                title="Top 10 Variables Négatives",
                labels={"SHAP Value": "", "Feature": ""},
                color="SHAP Value",
                color_continuous_scale="Reds"
            )
            fig_negative.update_coloraxes(showscale=False)

            # Afficher les graphiques côte à côte
            col1, col2 = st.columns(2)  # Diviser l'espace en deux colonnes
            with col1:
                st.plotly_chart(fig_positive, use_container_width=True)
            with col2:
                st.plotly_chart(fig_negative, use_container_width=True)

            
        st.markdown("---")  # Ligne de séparation
        
        
        
        # Sélection des variables pour une analyse bivariée
        st.subheader("Analyse de la distribution des variables")
        
        # Appeler la fonction pour récupérer dynamiquement les variables disponibles avec leurs types
        variables_disponibles = fetch_available_variables(client_id)

        # Extraire uniquement les noms des variables pour les afficher dans le multiselect
        variable_names = [var["name"] for var in variables_disponibles]

        # Sélectionner les variables pour afficher leurs distributions
        variables_a_afficher = st.multiselect(
            "",
            variable_names,  # Afficher uniquement les noms des variables
            default=[]  # Ne rien sélectionner par défaut
        )

        # Récupérer les distributions via l'API uniquement si des variables sont sélectionnées
        if variables_a_afficher:
            distribution_data = get_client_distribution(client_id, variables_a_afficher)

            # Vérifier si des données ont été récupérées
            if distribution_data:
                st.write(f"Distributions pour le client ID : {distribution_data['client_id']}")

                # Configurer les colonnes dynamiquement en fonction du nombre de variables sélectionnées
                cols = st.columns(len(variables_a_afficher))

                # Parcourir les variables sélectionnées
                for i, dist in enumerate(distribution_data["distributions"]):
                    variable = dist["variable"]
                    variable_type = dist.get("type", "numeric")  # Par défaut, traiter comme numérique
                    client_value = dist["client_value"]

                    # Gérer les variables numériques
                    if variable_type == "numeric":
                        global_values = dist["global_values"]

                        # Créer un histogramme interactif avec Plotly
                        fig = px.histogram(
                            global_values,
                            nbins=30,
                            title=f"Distribution de {variable}",
                            labels={"value": variable, "count": "Fréquence"},
                            opacity=0.75
                        )

                        # Ajouter une ligne verticale pour indiquer la position du client
                        fig.add_vline(
                            x=client_value,
                            line_width=3,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Valeur Client : {client_value:.0f}",
                            annotation_position="top left"
                        )

                        # Configurer le graphique
                        fig.update_layout(
                            xaxis_title=variable,
                            yaxis_title="Fréquence",
                            template="plotly_white",
                            height=400,
                            width=400
                        )

                    # Gérer les variables catégorielles
                    elif variable_type == "categorical":
                        category_counts = dist["category_counts"]

                        # Créer un diagramme en barres pour les variables catégorielles
                        fig = px.bar(
                            x=list(category_counts.keys()),
                            y=list(category_counts.values()),
                            title=f"Distribution de {variable}",
                            labels={"x": "Catégories", "y": "Fréquence"},
                            color=list(category_counts.keys())
                        )

                        # Ajouter une annotation pour la valeur du client
                        fig.add_annotation(
                            x=client_value,
                            y=category_counts.get(client_value, 0),
                            text=f"Valeur Client : {client_value}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="red"
                        )

                        # Configurer le graphique
                        fig.update_layout(
                            xaxis_title="Catégories",
                            yaxis_title="Fréquence",
                            template="plotly_white",
                            height=400,
                            width=400
                        )

                    # Afficher le graphique dans la colonne correspondante
                    with cols[i]:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée de distribution disponible pour les variables sélectionnées.")
        else:
            st.info("Aucune variable sélectionnée. Veuillez choisir des variables pour afficher leurs distributions.")





        
                
            
        # Sélection des variables pour une analyse bivariée
        st.subheader("Analyse Bivariée des Variables Numériques")

        # Récupérer dynamiquement les variables disponibles avec leurs types
        variables_disponibles = fetch_available_variables(client_id)

        # Filtrer pour inclure uniquement les variables numériques
        variables_numeriques = [
            var["name"] for var in variables_disponibles if var["type"] == "numeric"
        ]

        # Vérifier si des variables numériques sont disponibles
        if variables_numeriques:
            # Sélectionner les variables X et Y pour l'analyse bivariée
            col1, col2 = st.columns(2)
            with col1:
                variable_x = st.selectbox(
                    "Choisissez la variable X :",
                    ["Aucune"] + variables_numeriques  # Ajouter une option "Aucune"
                )
            with col2:
                variable_y = st.selectbox(
                    "Choisissez la variable Y :",
                    ["Aucune"] + variables_numeriques  # Ajouter une option "Aucune"
                )

            # Vérifier que les deux variables X et Y ont été sélectionnées
            if variable_x != "Aucune" and variable_y != "Aucune":
                # Récupérer les données de distribution pour les deux variables
                distribution_data = get_client_distribution(client_id, [variable_x, variable_y])

                if distribution_data:
                    # Récupérer les valeurs globales pour les deux variables
                    x_values = next(
                        dist["global_values"] for dist in distribution_data["distributions"] if dist["variable"] == variable_x
                    )
                    y_values = next(
                        dist["global_values"] for dist in distribution_data["distributions"] if dist["variable"] == variable_y
                    )

                    # S'assurer que les longueurs des listes sont cohérentes
                    min_length = min(len(x_values), len(y_values))
                    x_values = x_values[:min_length]
                    y_values = y_values[:min_length]

                    # Ajouter un graphique de dispersion pour visualiser la relation
                    fig = px.scatter(
                        x=x_values,
                        y=y_values,
                        labels={"x": variable_x, "y": variable_y},
                        title=f"Relation entre {variable_x} et {variable_y}",
                        opacity=0.75
                    )

                    # Ajouter la valeur du client sur le graphique
                    client_x = next(
                        dist["client_value"] for dist in distribution_data["distributions"] if dist["variable"] == variable_x
                    )
                    client_y = next(
                        dist["client_value"] for dist in distribution_data["distributions"] if dist["variable"] == variable_y
                    )
                    fig.add_scatter(
                        x=[client_x],
                        y=[client_y],
                        mode="markers+text",
                        marker=dict(color="red", size=10),
                        text=["Valeur Client"],
                        textposition="top center",
                        name="Client"
                    )

                    # Configurer et afficher le graphique
                    fig.update_layout(
                        template="plotly_white",
                        height=600,
                        width=800
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Impossible de récupérer les données pour l'analyse bivariée.")
            else:
                st.info("Veuillez sélectionner deux variables pour afficher leur relation.")
        else:
            st.warning("Aucune variable disponible pour l'analyse bivariée.")








st.markdown("---")  # Ligne de séparation



# Bouton pour afficher le graphique d'importance globale
if st.button("Afficher l'Importance Globale des Features"):
    st.warning("Veuillez noter que la génération de ce graphique peut prendre plusieurs minutes.")
    
    # Récupérer le graphique d'importance globale
    global_importance_graph = get_global_feature_importance_graph()

    if global_importance_graph:
        st.image(global_importance_graph, use_column_width=True, caption="Importance Globale des Features")
        
    # Récupérer et afficher le graphique SHAP
    shap_image = get_waterfall_graph(client_id)
    if shap_image:
        st.image(shap_image, use_column_width=True, caption="Graphique SHAP expliquant les variables influentes")
    else:
        st.warning("Impossible de charger le graphique SHAP.")



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







