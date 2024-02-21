# Déploiement de l'API sur Heroku via GitHub Actions

Déployer l'API et le Dashbord sur Heroku en utilisant GitHub Actions pour une intégration et un déploiement continu.

## Prérequis

Disposer des éléments suivants :

- Un compte Heroku.
- Un projet GitHub contenant votre code API et Dashboard.
- Des droits administratifs sur le projet GitHub pour configurer les actions et les secrets.

## Configuration des Secrets GitHub

Configurer les secrets nécessaires pour le déploiement :

1. Accédez à **Settings > Secrets > Actions** dans votre projet GitHub.
2. Ajoutez les secrets suivants :
    - `HEROKU_API_KEY`: Votre clé API Heroku, disponible dans les paramètres de votre compte Heroku.
    - `HEROKU_APP_NAME`: Le nom de votre application sur Heroku.
    - `HEROKU_EMAIL`: L'email associé à votre compte Heroku.

## Mise en Place du Workflow GitHub Actions

Votre workflow GitHub Actions, défini dans `.github/workflows/deploy.yml`, devrait ressembler à ceci :

```yaml
name: Deploy to Heroku

on:
push:
branches:
- main

jobs:
build:
runs-on: ubuntu-latest

      steps:
        - name: Checkout repository
          uses: actions/checkout@v4

        - name: Set up Python
          uses: actions/setup-python@v5
          with:
            python-version: 3.8

        - name: Install dependencies
          run: pip install -r requirements.txt

test :
needs: build
runs-on: ubuntu-latest

      steps:
        - name: Checkout repository
          uses: actions/checkout@v4

        - name: Set up Python
          uses: actions/setup-python@v5
          with:
            python-version: 3.8

        - name: Install dependencies
          run: pip install -r requirements.txt

        - name: Run tests
          run: pytest

deploy:
runs-on: ubuntu-latest

      steps:
        - name: Checkout repository
          uses: actions/checkout@v4

        - name: Deploy to Heroku
          uses: akhileshns/heroku-deploy@v3.13.15
          with:
            heroku_api_key: ${{secrets.HEROKU_API_KEY}}
            heroku_app_name:  ${{secrets.HEROKU_APP_NAME}}
            heroku_email: ${{secrets.HEROKU_EMAIL}}
```


## Vérification du Déploiement

Après le déploiement, vérifiez le statut de votre application via le tableau de bord Heroku.

## Support et Dépannage

Si vous rencontrez des problèmes lors du déploiement, voici quelques étapes à suivre pour le dépannage :

- **Vérifiez les Logs de Build sur GitHub Actions** : Accédez à l'onglet "Actions" de votre dépôt GitHub pour trouver les logs du workflow de déploiement.
- **Consultez les Logs d'Application sur Heroku** : Utilisez la commande `heroku logs --tail` dans votre terminal pour voir les logs.
- **Revérifiez les Secrets GitHub** : Assurez-vous que les secrets `HEROKU_API_KEY`, `HEROKU_APP_NAME`, et `HEROKU_EMAIL` sont correctement configurés dans les secrets de votre dépôt GitHub.

## Mises à jour et Maintenance

Pour mettre à jour votre application sur Heroku après des modifications :

1. Appliquez vos modifications dans le code de votre application.
2. Committez et pushez ces modifications sur la branche `main` de votre dépôt GitHub.
3. Le workflow GitHub Actions configuré se déclenchera automatiquement, procédant au build et au déploiement des mises à jour sur Heroku.

