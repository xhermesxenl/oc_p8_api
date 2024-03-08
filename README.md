# Projet 8 : Réalisez un dashboard

# API

## Description

Ce projet concerne le développement et le déploiement d'un Dashboard de scoring de crédit pour "Prêt à dépenser", une entreprise financière spécialisée dans l'octroi de crédits à la consommation pour les personnes ayant peu ou pas d'historique de crédit.
L'objectif principal du Dashboard  est de calculer automatiquement la probabilité de remboursement d'un crédit par un client et de classer la demande comme accordée ou refusée en fonction d'un seuil optimisé du point de vue métier.
Le Dashboard fournit une interface intuitive pour interagir avec l'API de scoring de crédit, permettant aux utilisateurs de tester et de visualiser les performances et les résultats du modèle de scoring.

## Architecture du Projet

Le dashboard utilise le framework Streamlit et appelle une API qui repose sur un modèle de scoring développé à partir de données comportementales et financières variées. 
Le projet suit une approche MLOps pour l'entraînement, le suivi, et le déploiement du modèle, en utilisant des outils tels que MLFlow pour le tracking des expérimentations, un registre centralisé des modèles, et GitHub Actions pour l'intégration et le déploiement continu.

Dépôt git du  projet Dashboard Interactif  : https://github.com/xhermesxenl/oc_p8_dashboard.git

## Découpage des Dossiers

- `/`: Code source pour l'entraînement du modèle, le déploiement de l'API, liste des packages.


## Installation

Pour configurer et exécuter l'API et le Dashboard localement, suivez ces étapes :

1. Clonez le dépôt GitHub.
2. Installez les dépendances en exécutant `pip install -r requirements.txt` dans le terminal.
3. Lancez l'API avec `python api.py`.

Pour configuer et exécuter le Dashboard localement, suivez les étapes dans le fichier README du repoertoire Streamlit

## Utilisation

L'API peut être testée localement via une requête HTTP POST avec un payload JSON contenant les données du client. Les instructions détaillées pour tester l'API sont disponibles dans `src/README.md`.

## Déploiement

Les instructions pour le déploiement de l'API et du Dashboard sur une plateforme cloud (Heroku) sont fournies dans le fichier `DEPLOYMENT.md`.

## Outils et Packages Utilisés

Un fichier `requirements.txt` est inclus, listant toutes les bibliothèques Python nécessaires pour exécuter l'API et le Dashboard.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)

## Contributions

Les contributions sont les bienvenues.
