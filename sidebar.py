import streamlit as st

def create_sidebar():

    # Titre de l'application
    st.sidebar.image("../logo.png", caption='logo', use_column_width=True)
    st.title("Prédiction du Risque de Non-Remboursement")

    st.sidebar.title("Saisir un numéro de client")

    # Créer une entrée de texte pour saisir l'ID du client et valider
    client_id_input = st.sidebar.text_input("Entrez l'ID du client (ex: 144194, 58369):")
    pressed = st.sidebar.button("Obtenir la prédiction")

    # Option d'affichage
    option_1 = st.sidebar.checkbox("Informations descriptives du client", False)
    option_2 = st.sidebar.checkbox("Facteurs d'influence locaux", False)
    option_3 = st.sidebar.checkbox("Facteurs d'influence globaux", False)
    option_4 = st.sidebar.checkbox("Comparaision aves l'ensemble des Clients ", False)
    option_5 = st.sidebar.checkbox("Comparaision aves un groupe de clients similaires", False)
    option_6 = st.sidebar.checkbox("Descriptions des caractéristiques", False)
    return pressed, option_1, option_2, option_3, option_4, option_5, option_6, client_id_input
