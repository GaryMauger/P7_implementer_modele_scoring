# Projet de Déploiement d’un Modèle de Scoring

## Objectif
Ce projet a été mené pour la société financière "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes avec peu ou pas d'historique de crédit. L'objectif principal est de développer un **modèle de scoring crédit** capable de prédire la probabilité de défaut de remboursement et d'assister les analystes financiers dans la prise de décision.

Le scoring est conçu pour améliorer l'**efficacité** et la **précision** des décisions de crédit, en se basant sur des **sources de données variées**, notamment des données comportementales et des informations provenant d'autres institutions financières.

Ce projet intègre une approche **MLOps** complète, couvrant la préparation des données, la modélisation, l'évaluation métier, l'analyse de la stabilité du modèle, et le déploiement en production, tout en mettant un accent particulier sur la **transparence** des décisions du modèle grâce à une analyse des caractéristiques influentes.

## Étapes du Projet
1. **Exploration des données** : Étude des données de clients et de leurs caractéristiques pour comprendre les facteurs liés aux défauts de paiement.
2. **Feature engineering** : Création de nouvelles variables pertinentes et transformations mathématiques pour enrichir les données disponibles.
3. **Modélisation** : Entraînement de plusieurs modèles (Dummy Classifier, Régression Logistique, Gradient Boosting) en utilisant des données déséquilibrées, puis application de **SMOTE** pour rééquilibrer les classes et améliorer la représentativité des clients défaillants.
4. **Optimisation des Hyperparamètres** : Ajustement des modèles **LGBM** et **XGBoost** via Randomized Search et validation croisée pour maximiser leurs performances.
5. **Ajout d'un Score Métier** : Définition d'une fonction de coût métier, qui pénalise davantage les faux négatifs (erreurs de prédiction coûteuses), et optimisation des seuils de décision en fonction du contexte métier.
6. **Analyse des Caractéristiques (Feature Importance)** :
   - **Globale** : Identifier les caractéristiques qui influencent le plus le modèle au niveau général.
   - **Locale** : Utiliser les valeurs SHAP pour expliquer les prédictions à un niveau individuel, permettant une transparence dans l'attribution des scores aux clients.
7. **Analyse de la Dérive des Données (Data Drift)** : Surveillance des changements dans les distributions des données en production pour s'assurer de la robustesse et de la validité continue du modèle.
8. **Déploiement de l'API** :
   - Développement d'une **API avec FastAPI** pour permettre l'intégration du modèle avec les systèmes existants.
   - Conteneurisation avec **Docker** pour faciliter le déploiement sur des environnements divers.
   - Mise en place d'une **interface utilisateur avec Streamlit**, hébergée sur Streamlit Cloud, pour tester les prédictions du modèle de manière interactive.

## Structure des Dossiers et Fichiers
- **Notebook `Projet_7_1.ipynb`** : Exploration des données, vérification de la qualité des données, transformations (normalisation, encodage), et feature engineering.
- **Notebook `Projet_7_2.ipynb`** : Entraînement des modèles, application du SMOTE, optimisation des hyperparamètres, ajout du score métier, et analyse des performances.
- **Dossier `Data`** :
  - Contient les données du projet ([Source Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)).
  - Dossier contenant le modèle entraîné.
- **Dossier `API`** :
  - `api.py` : API de prédiction réalisée avec **FastAPI**.
  - `model.py` : Fonctions de prédiction utilisées par l'API.
  - `pipeline_features_eng.py` : Transformations des données.
  - `Dockerfile` : Fichier pour la **conteneurisation de l'API**.
- **Dossier `app_streamlit`** :
  - `app.py` : Tableau de bord interactif avec **Streamlit**.
  - `Dockerfile` : Fichier pour la **conteneurisation de l'application**.
  - `logo_pret_a_depenser.png` : Logo utilisé dans le tableau de bord.
- **`requirements.txt`** : Packages nécessaires pour le déploiement.
- **`unit_tests.py`** : Tests unitaires pour l'API (intégrés au CI/CD).
- **`data_drift_analysis.html`** : Rapport d'analyse de dérive des données en production.

    ```

## Historique des Modifications
Le projet est versionné à l'aide de **Git**. Plusieurs versions ont été publiées sur GitHub pour suivre l'évolution du développement.

