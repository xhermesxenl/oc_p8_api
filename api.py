import pandas as pd
import pickle
import shap
import os
import numpy as np
import json

from sklearn.preprocessing import MinMaxScaler
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
api = Api(app, version='1.0', title='API de Score de Crédit',
          description='Une API permettant de calculer et de gérer les scores de crédit')

# Définition du namespace
ns = api.namespace('api', description='Opérations principales')
api.add_namespace(ns, path='/api')

model_filename = "model.pkl"
data_app = pd.read_csv("data.csv")
data_info = pd.read_csv("data_info.csv")
data_global = pd.read_csv("feature_importances.csv")
data_desc = pd.read_csv("HomeCredit_columns_description.csv", encoding="ISO-8859-1")
data_info_j = json.loads(data_info.to_json())
data_app_j = json.loads(data_app.to_json())

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
def minmax(data):
    col = data['TARGET']
    cols_to_exclude = ['TARGET', 'SK_ID_CURR']
    df = data.select_dtypes(include='number').drop(cols_to_exclude, axis=1)

    scaler = MinMaxScaler().fit(df)
    df_minmax = pd.DataFrame(scaler.transform(df), columns=df.columns,
                               index=df.index)

    df_minmax['TARGET'] = col
    df_minmax['SK_ID_CURR'] = data['SK_ID_CURR']
    return df_minmax

df_minmax = minmax(data_app)


def voisins_proches_id(client_id, df_minmax, n_neighbors=10):
    """
    Trouve les voisins les plus proches pour un client donné
    """

    print("start")
    # Vérifier si le client existe dans les données
    # if client_id not in df_minmax['SK_ID_CURR']:
    #     raise ValueError("Le client avec l'identifiant {} n'existe pas dans les données.".format(client_id))
    if client_id in df_minmax["SK_ID_CURR"].tolist():
        print("id_client ok in dataminmax")

    feat = df_minmax.loc[df_minmax["SK_ID_CURR"] == client_id].drop(columns=["TARGET", "SK_ID_CURR"]).values.reshape(1, -1)
    X_train = df_minmax.drop(columns=["TARGET", "SK_ID_CURR"]).values

    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(X_train)

    indices = knn.kneighbors(feat, return_distance=False)

    similar_clients = data_info.iloc[indices[0], :]
    similar_clients_norm = df_minmax.iloc[indices[0], :]

    return similar_clients, similar_clients_norm


@ns.route('/')
class Welcome(Resource):
    def get(self):
        return "Bienvenue dans l'API de prédiction de crédit !"

@ns.route('/predict/<int:id_client>')
class PredictScoreClient(Resource):
    def get(self, id_client):
        id_client = int(id_client)
        if id_client in data_app["SK_ID_CURR"].tolist():
            data_client = data_app.loc[data_app["SK_ID_CURR"] == id_client]
            data_client = data_client.drop(["SK_ID_CURR", "TARGET"], axis=1)
            proba = model.predict_proba(data_client)

            proba_1 = round(proba[0][1] * 100)

            seuil_optimal = 0.48
            value_seuil_optimal =   seuil_optimal * 100

            # Classer le client comme "accepté" ou "refusé" en fonction du seuil
            classe = "refuse" if proba_1 > value_seuil_optimal else "accepte"

            return jsonify({ "probability": proba_1, "classe": classe})
        else:
            return jsonify({"error": "Unknown ID"}), 404

@ns.route('/predictpost')
class PredictScoreClientPost(Resource):
    def post(self):
        request_data = request.get_json()

        id_client = request_data.get('id_client')
        id_client = int(id_client)
        updated_data = request_data.get('data')

        if id_client and id_client in data_app["SK_ID_CURR"].tolist():

            # Extraction et mise à jour des données client
            data_client = data_app.loc[data_app["SK_ID_CURR"] == id_client]
            data_client = data_client.drop(["SK_ID_CURR", "TARGET"], axis=1)

            # Mettre à jour data_client avec updated_data ici
            for key, value in updated_data.items():
                print(key, value)
                if key in data_client.columns:
                    data_client[key] = float(value)

            # Prédiction
            proba = model.predict_proba(data_client)
            proba_1 = round(proba[0][1] * 100)
            seuil_optimal = 0.48
            value_seuil_optimal = seuil_optimal * 100
            classe = "refusé" if proba_1 > value_seuil_optimal else "accepté"

            return jsonify({"probability": proba_1, "classe": classe})
        else:
            return jsonify({"error": "Unknown ID"}), 404

@ns.route('/data/<int:id_client>')
class InfoClient(Resource):
    def get(self, id_client):
        if id_client in data_info["SK_ID_CURR"].tolist():
            data_client = data_info.loc[data_info["SK_ID_CURR"] == id_client]
            return jsonify(data_client.to_dict(orient="records"))
        else:
            return jsonify({"error": "Unknown ID"}), 404

"""Retourne les informations descriptives pour tous les clients."""
@ns.route('/data/all')
class AllClient(Resource):
    def get(self):
        try:
            data_j = json.loads(df_minmax.to_json())

            return jsonify({
                'data': data_j,
                'data_test': data_test_j,
                'data_info': data_info_j,
            }), 200
        except Exception as e:
            return jsonify({'erreur': str(e)}), 500

### Comparaison Knn
@ns.route('/data/knn/<int:id_client>')
class KnnDataClient(Resource):
    def get(self, id_client):

        try:
            id_client = int(id_client)
            sim_clients, sim_clients_norm = voisins_proches_id(id_client, df_minmax)

            # df_sim_client_j = json.loads(sim_clients.to_json())
            # df_sim_client_norm_j = json.loads(sim_clients_norm.to_json())

            df_sim_client_j = json.loads(sim_clients.to_json(orient='records'))
            df_sim_client_norm_j = json.loads(sim_clients_norm.to_json(orient='records'))

            return jsonify({"df_sim_client": df_sim_client_j,
                            "df_sim_client_norm": df_sim_client_norm_j})
        except Exception as e:
            return jsonify({'erreur': str(e)}), 500


### Feature Local
@ns.route('/shap/<int:id_client>')
class ShapValuesClient(Resource):
    def get(self, id_client):
        id_client = int(id_client)
        if id_client in data_app["SK_ID_CURR"].tolist():
            data_client = data_app.loc[data_app["SK_ID_CURR"] == id_client]
            data_client = data_client.drop(["SK_ID_CURR","TARGET"], axis=1)
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
@ns.route('/global')
class FeatureGlobal(Resource):
    def get(self):
        feat_glob = json.loads(data_global.to_json())
        return jsonify({'feat_imp_global': feat_glob})

### Desc All
@ns.route('/desc/all')
class FeatureLocal(Resource):
    def get(self):
        data = json.loads(data_desc.to_json())
        return jsonify({'desc': data})

if __name__ == "__main__":
  app.run(debug=True)








