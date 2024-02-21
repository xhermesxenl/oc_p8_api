import os
import pickle
import pytest

from api import model_path

model_filename = "model.pkl"
model_path = os.path.join(os.getcwd(), model_filename) 


def test_model_file_exists():
    """Test pour vérifier si le fichier du modèle existe."""
    assert os.path.isfile(model_path), f"Le fichier modèle '{model_path}' n'existe pas."


def test_model_loads_correctly():
    """Test pour vérifier que le modèle se charge correctement."""
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        assert model is not None, "Le modèle n'a pas pu être chargé."
    else:
        pytest.skip("Le fichier modèle n'existe pas, le test de chargement est ignoré.")

