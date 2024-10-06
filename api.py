import mlflow
import mlflow.lightgbm
from fastapi import FastAPI

# Initialiser l'application FastAPI
app = FastAPI()

# Spécifier le chemin du modèle
model_path = "file:///C:/Users/mauge/Documents/github/P7_implementer_modele_scoring/mlartifacts/404984096885784552/4deefe68adea4a65a11fb225abed04a6/artifacts/model"

# Charger le modèle à partir du chemin local
model = mlflow.lightgbm.load_model(model_path)

@app.get("/")
def great():
    return {"message": "Modèle chargé avec succès"}


