import pytest
from fastapi.testclient import TestClient
from api import app  # Importation de l'application FastAPI

# Créer un client de test pour FastAPI
client = TestClient(app)

# Test pour la route d'accueil
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de prédiction avec FastAPI"}

# Test pour la route de prédiction
@pytest.mark.parametrize("client_id, expected_status", [(111761, 200), (999999, 404)])
def test_predict_client(client_id, expected_status):
    response = client.post("/predict", json={"client_id": client_id})
    assert response.status_code == expected_status
    if expected_status == 200:
        assert "probability" in response.json()
        assert "prediction" in response.json()

# Test pour la route /waterfall/{client_id}
@pytest.mark.parametrize("client_id, expected_status", [(111761, 200), (999999, 404)])
def test_waterfall(client_id, expected_status):
    response = client.get(f"/waterfall/{client_id}")
    assert response.status_code == expected_status
    if expected_status == 200:
        assert "waterfall_image" in response.json()

# Test pour la route de l'importance des features globales
def test_global_feature_importance():
    response = client.get("/global_feature_importance/")
    assert response.status_code == 200
    assert "image" in response.json()

# Test pour la liste des clients
def test_get_clients():
    response = client.get("/clients")
    assert response.status_code == 200
    assert "client_ids" in response.json()
    assert isinstance(response.json()["client_ids"], list)