import streamlit as st
from multiapp import MultiApp
from apps import home, model, visualisation, word2vec


app = MultiApp()

st.set_page_config(layout="wide")

st.title('Exploration des Tweets liés à la Superleague Européenne')


# Add all your application here
app.add_app("Présentation du Projet", home.app)
app.add_app("Visualisation", visualisation.app)
app.add_app("Topic Modeling", model.app)
app.add_app("Word Embeddings", word2vec.app)
# The main app
app.run()