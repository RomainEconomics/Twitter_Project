import streamlit as st

def app():
    st.title('Présentation du projet text mining')

    st.markdown("""
    Cette application permet de visualiser les résulats de modèles de NLP comme le modèle LDA, le modèle NMF ou bien la méthode de Word Embeddings.
    * **Python librairies:** Scikit-learn, Pandas, Streamlit
    * **NLP Librairies:** Spacy, Gensim
    * **Data source:** Tweets extrait à partir de la librairie Twint.
    """)