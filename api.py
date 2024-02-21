import pandas as pd
import pickle
import shap
import os
import numpy as np
import json

from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify, render_template
from flask_restx import Api, Resource, Namespace, fields

from shap import LinearExplainer, KernelExplainer, Explanation, TreeExplainer
from shap.maskers import Independent
shap.initjs()

# Exemple
# http://127.0.0.1:5000/api/predict/124782
# http://127.0.0.1:5000/api/predict/58369

app = Flask(__name__)
api = Api(app, version='1.0', title='API Example avec Flask-RESTPlus',
          description='Une simple API documentée avec Flask-RESTPlus')

# Définition du namespace
ns = api.namespace('api', description='Opérations principales')
api.add_namespace(ns, path='/api')

model_filename = "model.pkl"
data_train = pd.read_csv("train_api.csv")
data_test = pd.read_csv("test_api.csv")
data_info = pd.read_csv("data_info.csv")
data_global = pd.read_csv("feature_importances.csv")
data_desc = pd.read_csv("HomeCredit_columns_description.csv", encoding="ISO-8859-1")
data_json = json.loads(data_info.to_json())
data_train_j = json.loads(data_train.to_json())
data_test_j = json.loads(data_test.to_json())

ids = list(data_test.index.to_series())


model_path = os.path.join(os.getcwd(), model_filename)

# Charger le modèle depuis un fichier
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

lgbm_model = model.named_steps["LGBMC"]
explainer = shap.TreeExplainer(lgbm_model)

treshold = 0.48

                        #######################
                        ###      KNN        ###
                        #######################

def voisins_proches(data_train, n_neighbors=5):

    # Créez et entraînez le modèle des plus proches voisins
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    neighbors = knn.fit(data_train)

    return neighbors



def voisins_proches_id(df_test, data_train, client_id, n_neighbors=5):
    """
    Trouve les voisins les plus proches pour un client donné dans le jeu de données d'entraînement.

    :param client_id: Identifiant unique du client dans le jeu de données de test.
    :param n_neighbors: Nombre de voisins les plus proches à trouver.
    :return: DataFrame contenant les informations des voisins similaires.
    """
    test_point = df_test.loc[[client_id]].values

    # Créez et entraînez le modèle des plus proches voisins
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    neighbors = knn.fit(data_train)

    # Trouvez les voisins les plus proches
    distances, indices = knn.kneighbors(test_point)

    # Récupérez les voisins similaires à partir de data_train
    similar_clients = data_train.iloc[indices[0]]
    return similar_clients

@ns.route('/')
class Welcome(Resource):
    def get(self):
        return "Bienvenue dans l'API de prediction de credit !"

@ns.route('/ids')
class ClientsIDs(Resource):
    def get(self):
    # Return en json format
        return jsonify({'status': 'ok',
                        'data': ids})



@ns.route('/predict/<int:id_client>')
class PredictScoreClient(Resource):
    def get(self, id_client):
        if id_client in data_test["SK_ID_CURR"].tolist():
            data_client = data_test.loc[data_test["SK_ID_CURR"] == id_client]
            data_client = data_client.drop(["Unnamed: 0", "SK_ID_CURR"], axis=1)
            proba = model.predict_proba(data_client)

            proba_1 = round(proba[0][1] * 100)

            seuil_optimal = 0.48
            value_seuil_optimal =   seuil_optimal * 100

            # Classer le client comme "accepté" ou "refusé" en fonction du seuil
            classe = "refuse" if proba_1 > value_seuil_optimal else "accepte"

            return jsonify({ "probability": proba_1, "classe": classe})
        else:
            return jsonify({"error": "Unknown ID"}), 404



@app.get("/api/data/<int:id_client>/")
def donnees_client(id_client):
    if id_client in data_info["SK_ID_CURR"].tolist():
        data_client = data_info.loc[data_info["SK_ID_CURR"] == id_client]
        return jsonify(data_client.to_dict(orient="records"))
    else:
        return jsonify({"error": "Unknown ID"}), 404

    # client_model = ns.model('Client', {
    #     'SK_ID_CURR': fields.Integer(description='Identifiant du client')
    # })

#
# @ns.route('/data/<int:id_client>/')
# @ns.response(404, 'Identifiant non trouvé')
# @ns.param('id_client', 'L\'identifiant du client')
# class DonneesClient(Resource):
#     @ns.marshal_with(client_model, envelope='data', code=200, skip_none=True)
#     def get(self, id_client):
#         """Retourne les données du client par ID"""
#         if id_client in data_info["SK_ID_CURR"].tolist():
#             data_client = data_info.loc[data_info["SK_ID_CURR"] == id_client]
#             # Conversion du DataFrame en dictionnaire pour la réponse
#             return data_client.to_dict(orient="records")[0]
#         else:
#             ns.abort(404, "Unknown ID")

"""Retourne les informations descriptives pour tous les clients."""
@app.get('/api/data/all')
def tous_data_clients():
    try:
        return jsonify({
            'data_train': data_train_j,
            'data_test': data_test_j
        }), 200
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500

"""Retourne les informations descriptives pour tous les clients."""

@app.get('/api/data/knn/<int:id_client>')
def knn_data_clients():
    try:

        return jsonify({
            'data_train': data_train_j,
            'data_test': data_test_j
        }), 200
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500

@ns.route('/shap/<int:id_client>')
class ShapValuesClient(Resource):
    def get(self, id_client):
        print("shap")
        if id_client in data_test["SK_ID_CURR"].tolist():
            data_client = data_test.loc[data_test["SK_ID_CURR"] == id_client]
            data_client = data_client.drop(["Unnamed: 0", "SK_ID_CURR"], axis=1)
            data_client_array = data_client.values.reshape(1, -1)

            # Calculer les valeurs SHAP
            shap_values = explainer.shap_values(data_client_array)

            # Obtenir la valeur de base (expected value) et les noms des caractéristiques
            expected_value = explainer.expected_value
            feature_names = list(data_client.columns)

            # Préparez les données pour la réponse
            response_data = {
                # "shap_values": shap_values[0].tolist(),
                # "expected_value": expected_value.tolist() if isinstance(expected_value, np.ndarray) else expected_value,
                # "feature_names": feature_names
                "shap_values_class_0": shap_values[0][0].tolist(),  # Le résultat de la première dimension/classe
                "shap_values_class_1": shap_values[1][0].tolist(),  # Le résultat de la deuxième dimension/classe
                "expected_value_class_0": expected_value[0] if isinstance(expected_value, list) else expected_value,
                "expected_value_class_1": expected_value[1] if isinstance(expected_value, list) else expected_value,
                "feature_names": feature_names
            }
            return jsonify(response_data)
        else:
            return jsonify({"error": "Unknown ID"}), 404

### Feature Global
@app.get("/api/global")
def get_feature_importances():
    feat_glob = json.loads(data_global.to_json())
    return jsonify({'feat_imp_global': feat_glob})

### Feature Global
@app.get("/api/desc/all")
def get_desc():
    data = json.loads(data_desc.to_json())
    return jsonify({'desc': data})



if __name__ == "__main__":
  app.run(debug=True)








