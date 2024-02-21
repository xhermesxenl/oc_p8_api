import pytest
from flask.testing import FlaskClient
from api import app  # Assurez-vous que ce chemin est correct

# Configuration du client de test Flask
@pytest.fixture
def client() -> FlaskClient:
    with app.test_client() as client:
        yield client

# Test de la route racine          
def test_welcome(client):
    response = client.get("/api/")
    assert response.status_code == 200
    assert "Bienvenue dans l'API de prediction de credit !" in response.data.decode()

# Test de la route de prédiction avec une requête POST valide
def test_predict_credit_valid(client):

    id_accept = 144194

    response = client.get(f"/api/predict/{id_accept}")
    assert response.status_code == 200
    assert "probability" in response.json
    assert "classe" in response.json
    assert response.json["classe"] in ["accepte"]


# Test de la route de prédiction avec une requête POST invalide
def test_predict_credit_invalid(client):

    id_refuse = 13112
    response = client.get(f"/api/predict/{id_refuse}")
    assert response.status_code == 200
    assert "probability" in response.json
    assert "classe" in response.json
    assert response.json["classe"] in ["refuse"]


