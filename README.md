# Projet de Déploiement d’un Modèle de Scoring

## Objectif
Ce projet a pour but de développer un modèle de scoring crédit pour prédire la probabilité de défaut de remboursement des clients. Le modèle est déployé via une API et un tableau de bord interactif, permettant une utilisation pratique et évolutive.

## Structure des dossiers et fichiers

- **Notebook `Projet_7_1.ipynb`** : Code pour la préparation des données et l'analyse exploratoire.
- **Notebook `Projet_7_2.ipynb`** : Code pour les essais de modélisation.
- **Dossier `Data`** :
  - Contient les données du projet ([Source Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)).
  - Dossier contenant le modèle entraîné.
  
- **Dossier `API`** :
  - `api.py` : API de prédiction réalisée avec FastAPI.
  - `model.py` : Fonctions de prédiction utilisées par l'API.
  - `pipeline_features_eng.py` : Transformations des données.
  - `Dockerfile` : Fichier pour la containerisation de l'API.
  
- **Dossier `app_streamlit`** :
  - `app.py` : Tableau de bord interactif avec Streamlit.
  - `Dockerfile` : Containerisation de l'application.
  - `logo_pret_a_depenser.png` : Logo utilisé dans le tableau de bord.
  
- **`requirements.txt`** : Packages nécessaires pour le déploiement.
- **`unit_tests.py`** : Tests unitaires pour l'API (CI/CD).
- **`data_drift_analysis.html`** : Rapport d'analyse de dérive des données.

## Installation et exécution

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/projet-scoring.git

 
