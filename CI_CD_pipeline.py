import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

# Chargement des encodeurs depuis le fichier
encoders = joblib.load('C:/Users/mauge\Documents/github/P7_implementer_modele_scoring/label_encoders.pkl')
df_data_5 = pd.read_pickle('C:/Users/mauge\Documents/github/P7_implementer_modele_scoring/encoded_features.pkl')

# Fonction pour charger les DataFrames
def load_data():
    # Charger les DataFrames à partir des fichiers CSV ou d'autres sources
    df_application_test = pd.read_csv('C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')
    df_application_train = pd.read_csv('C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv')
    
    # Concaténer les datasets application_train et application_test
    df_data = pd.concat([df_application_train, df_application_test], axis=0)
    
    # Charger les autres datasets
    df_previous_application = pd.read_csv('C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Projet+Mise+en+prod+-+home-credit-default-risk/previous_application.csv')
    df_credit_card_balance = pd.read_csv('C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Projet+Mise+en+prod+-+home-credit-default-risk/credit_card_balance.csv')
    df_installments_payments = pd.read_csv('C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Projet+Mise+en+prod+-+home-credit-default-risk/installments_payments.csv')
    df_POS_CASH_balance = pd.read_csv('C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Projet+Mise+en+prod+-+home-credit-default-risk/POS_CASH_balance.csv')
    
    return {
        'df_application_train': df_application_train,
        'df_application_test': df_application_test,
        'df_data': df_data,
        'df_previous_application': df_previous_application,
        'df_credit_card_balance': df_credit_card_balance,
        'df_installments_payments': df_installments_payments,
        'df_POS_CASH_balance': df_POS_CASH_balance
    }

# Fonction pour préparer les jointures sur df_previous_application
def prepare_aggregations(df_data, df_previous_application, df_credit_card_balance, df_installments_payments, df_POS_CASH_balance):
    # Jointure et agrégations sur les différentes tables
    previous_application_counts = df_previous_application.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(columns={'SK_ID_PREV': 'PREVIOUS_APPLICATION_COUNT'})
    df_data = df_data.merge(previous_application_counts, on='SK_ID_CURR', how='left')

    # Moyennes pour credit card balance
    credit_card_balance_mean = extract_mean(df_credit_card_balance, 'CARD_MEAN_')
    previous_application = df_previous_application.merge(credit_card_balance_mean, on='SK_ID_PREV', how='left')

    # Moyennes pour installments payments
    install_pay_mean = extract_mean(df_installments_payments, 'INSTALL_MEAN_')
    previous_application = previous_application.merge(install_pay_mean, on='SK_ID_PREV', how='left')

    # Moyennes pour POS_CASH_balance
    POS_mean = extract_mean(df_POS_CASH_balance, 'POS_MEAN_')
    previous_application = previous_application.merge(POS_mean, on='SK_ID_PREV', how='left')

    # Moyenne des colonnes numériques pour previous_application
    prev_appl_mean = extract_mean(previous_application, 'PREV_APPL_MEAN_', group_by='SK_ID_CURR')
    prev_appl_mean = prev_appl_mean.rename(columns={'PREV_APPL_MEAN_SK_ID_CURR': 'SK_ID_CURR'})

    # Fusionner avec df_data
    df_data = df_data.merge(prev_appl_mean, on='SK_ID_CURR', how='left')

    return df_data

# Fonction pour extraire les moyennes par groupe
def extract_mean(df, prefix, group_by='SK_ID_PREV'):
    numeric_cols = df.select_dtypes(include='number').copy()
    numeric_cols[group_by] = df[group_by]
    y = numeric_cols.groupby(group_by, as_index=False).mean().add_prefix(prefix)
    y = y.rename(columns={f'{prefix}{group_by}': group_by})
    return y

# Fonction pour créer des features polynomiales
def create_polynomial_features(df_data, degree=3):
    # Sélectionner les colonnes pour les features polynomiales
    poly_features = df_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    # Imputer les valeurs manquantes
    imputer = SimpleImputer(strategy='median')
    poly_features = imputer.fit_transform(poly_features)

    # Créer l'objet PolynomialFeatures
    poly_transformer = PolynomialFeatures(degree=degree)

    # Transformer les caractéristiques
    poly_features = poly_transformer.fit_transform(poly_features)

    # Obtenir les noms des nouvelles features créées
    feature_names = poly_transformer.get_feature_names_out(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])

    # Créer un DataFrame pour les caractéristiques transformées
    poly_features_df = pd.DataFrame(poly_features, columns=feature_names)

    # Ajouter la colonne cible et SK_ID_CURR au DataFrame de caractéristiques polynomiales
    poly_features_df['SK_ID_CURR'] = df_data['SK_ID_CURR']

    # Supprimer les colonnes d'origine pour éviter les doublons
    columns_to_drop = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
    poly_features_df = poly_features_df.drop(columns=[col for col in columns_to_drop if col in poly_features_df.columns])

    # Fusionner avec df_data
    df_data_merged = df_data.merge(poly_features_df, on='SK_ID_CURR', how='left')

    return df_data_merged

# Fonction pour préparer les variables métier
def create_business_features(df):
    # Imputation des valeurs manquantes avant de créer de nouvelles variables
    imputer = SimpleImputer(strategy='median')
    
    # Appliquer l'imputation sur les colonnes nécessaires
    df[['AMT_ANNUITY', 'DAYS_EMPLOYED']] = imputer.fit_transform(df[['AMT_ANNUITY', 'DAYS_EMPLOYED']])

    # Création des nouvelles variables métier
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Conserver uniquement les colonnes d'intérêt et les identifiants
    business_features = df[['SK_ID_CURR', 'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 
                            'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']]
    
    return business_features

def encode_features(df, encoders):
    df_encoded = df.copy()

    # Encodage des colonnes avec LabelEncoder
    for col, le in encoders.items():
        if col in df_encoded.columns:
            df_encoded[col] = le.transform(df_encoded[col])
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_encoded, drop_first=True)

    # Assurez-vous que toutes les colonnes nécessaires sont présentes
    for col in encoders.keys():
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Ajoutez la colonne manquante avec des zéros
    
    # Réorganisez les colonnes pour correspondre à df_data_5
    df_final = df_encoded.reindex(columns=df_data_5.columns, fill_value=0)

    return df_final

def handle_missing_values(df):
    # Initialisation de l'imputateur avec la stratégie médiane
    imputer = SimpleImputer(strategy='median')
    
    # Vérification du nombre de valeurs manquantes avant l'imputation
    missing_values_before = df.isnull().sum().sum()
    
    # Apprentissage de l'imputateur et transformation des données
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Réaffectation des index originaux
    df_imputed.index = df.index
    
    # Vérification du nombre de valeurs manquantes après l'imputation
    missing_values_after = df_imputed.isnull().sum().sum()
    
    return df_imputed

def execute_pipeline():

# Charger les données
    df_data, df_previous_application, df_credit_card_balance, df_installments_payments, df_POS_CASH_balance = load_data()
    
    # Préparer les données agrégées (supposons qu'une fonction 'prepare_aggregations' existe)
    df_data_prepared = prepare_aggregations(df_data, df_previous_application, df_credit_card_balance, df_installments_payments, df_POS_CASH_balance)

    # Créer les features métier
    business_features = create_business_features(df_data_prepared)

    # Créer les features polynomiales
    df_final = create_polynomial_features(df_data_prepared)

    # Supprimer les colonnes existantes dans df_final qui vont être remplacées par celles des business_features
    cols_to_replace = ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']
    df_final = df_final.drop(columns=cols_to_replace, errors='ignore')

    # Fusionner les variables métier avec les features polynomiales, sans suffixes
    df_final = df_final.merge(business_features, on='SK_ID_CURR', how='left')

    # Encodage des variables catégorielles
    df_final = encode_features(df_final, encoders)

    # Gestion des valeurs manquantes
    df_final = handle_missing_values(df_final)

    # Afficher les informations finales
    #print('Shape of final data:', df_final.shape)
    return df_final

# Exécuter le pipeline
df_data_6 = execute_pipeline()

# Fonction principale pour séparer les ensembles d'entraînement et de validation
def prepare_train_test_split(df_data_6, data):
    # Récupérer les DataFrames depuis le dictionnaire data
    df_application_train = data['df_application_train']
    df_application_test = data['df_application_test']

    # Séparer les données en train et test en fonction de SK_ID_CURR
    df_train = df_data_6[df_data_6['SK_ID_CURR'].isin(df_application_train['SK_ID_CURR'])]
    df_test = df_data_6[df_data_6['SK_ID_CURR'].isin(df_application_test['SK_ID_CURR'])]

    # Colonnes à exclure de la standardisation
    exclude_cols = ['SK_ID_CURR', 'TARGET']
    
    # Séparation des colonnes à standardiser de celles à exclure
    cols_to_scale = [col for col in df_train.columns if col not in exclude_cols]
    df_train_to_scale = df_train[cols_to_scale]
    df_train_excluded = df_train[exclude_cols]
    
    # Création et application du scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train_scaled = scaler.fit_transform(df_train_to_scale)
    
    # Recréer un DataFrame avec les colonnes standardisées
    df_train_scaled_df = pd.DataFrame(df_train_scaled, columns=cols_to_scale)
    
    # Ajouter les colonnes exclues à nouveau après la standardisation
    train = pd.concat([df_train_scaled_df, df_train_excluded.reset_index(drop=True)], axis=1)

    print('Standardization complete.')
    print('Data shape after scaling: ', train.shape)

    # Séparation des features (X) et de la target (y)
    X_train = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y_train = train['TARGET']

    # Séparer l'ensemble d'entraînement en un ensemble d'entraînement et un ensemble de validation
    X_train_split, X_val, y_train_split, y_val = tts(X_train, y_train, test_size=0.3, random_state=42)
    
    return X_train_split, X_val, y_train_split, y_val

# Exemple d'utilisation de la fonction
best_params_xgb = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'gamma': 0.1,
    'min_child_weight': 5,
    'tree_method': 'gpu_hist',  # Option pour utiliser le GPU si disponible
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42
}

def train_and_evaluate_xgb(X_train_split, y_train_split, X_val, y_val, best_params):
    """
    Fonction pour entraîner un modèle XGBoost avec les hyperparamètres donnés et évaluer ses performances.

    Args:
    - X_train_split: Les features d'entraînement.
    - y_train_split: Les labels d'entraînement.
    - X_val: Les features de validation.
    - y_val: Les labels de validation.
    - best_params: Dictionnaire des hyperparamètres optimaux.

    Returns:
    - model: Le modèle XGBoost entraîné.
    - metrics: Dictionnaire contenant les scores d'accuracy et de ROC AUC sur l'ensemble de validation.
    """
    
    # Création du modèle XGBoost avec les hyperparamètres optimaux
    model_xgb = xgb.XGBClassifier(**best_params)

    # Entraînement du modèle
    print("Entraînement du modèle XGBoost...")
    model_xgb.fit(X_train_split, y_train_split)

    # Prédictions sur l'ensemble de validation
    y_val_pred = model_xgb.predict(X_val)
    y_val_pred_proba = model_xgb.predict_proba(X_val)[:, 1]

    # Calcul des métriques
    accuracy = accuracy_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred_proba)

    # Affichage des résultats
    print(f"Accuracy sur l'ensemble de validation : {accuracy:.4f}")
    print(f"ROC AUC sur l'ensemble de validation : {roc_auc:.4f}")
    
    # Retourner le modèle et les métriques
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    return model_xgb, metrics

# Entraînement et évaluation du modèle
model_xgb, metrics = train_and_evaluate_xgb(X_train_split, y_train_split, X_val, y_val, best_params_xgb)








df_clients = df_final

data_path = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/"

df_application_test = pd.read_csv(data_path + 'Projet+Mise+en+prod+-+home-credit-default-risk/' + 'application_test.csv')

df_clients = df_clients[df_clients['SK_ID_CURR'].isin(df_application_test['SK_ID_CURR'])]

# Échantillonner 5000 lignes aléatoirement (assurez-vous que le DataFrame contient au moins 5000 lignes)
df_clients = df_clients.sample(n=5000, random_state=42)

# Créer un df_application_test_sample en utilisant les mêmes SK_ID_CURR que df_clients_sample
df_application_test = df_application_test[df_application_test['SK_ID_CURR'].isin(df_clients['SK_ID_CURR'])]

# Chemin où enregistrer le fichier localement
output_path = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Data/df_clients.csv"
output_path_application = "C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/Data/df_application_test.csv"

# Enregistrer le DataFrame en CSV (sans compression ici pour faciliter l'utilisation sur GitHub)
df_clients.to_csv(output_path, index=False)
df_application_test.to_csv(output_path_application, index=False)