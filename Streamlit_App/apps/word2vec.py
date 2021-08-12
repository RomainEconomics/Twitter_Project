import streamlit as st
from gensim.models import Word2Vec
import pandas as pd

def app():
    st.header("Utilisation de l'algorithme Word2Vec")
    st.markdown("""
    Ici, nous utilisons une nouvelle librairie (Gensim) pour y effectuer une réprésentation 
    vectorielle de notre corpus de documents via l'utilisation de l'algorithme Word2Vec. 
    Avec cette nouvelle représentation de nos données textuelles, le sens d'un mot est finalement 
    un point dans un espace vectoriel basé sur la distribution de chaque mot. 
    Ainsi, cela nous permet de représenter des mots ayant un sens similaire dans un espace vectoriel proche. 

    Plutot que de compter le nombre d'occurences de chaque mot
    à proximité d'un autre, on cherche à prédire si un mot a une plus ou moins forte probabilité de se 
    trouver à proximité d'un autre mot.

    Néanmoins, comme cet algorithme permet de représenter nos données sous la forme d'une matrice avec 
    100 colonnes, il est nécessaire d'utiliser un algorithme de réduction de dimensions.
    L'algorithme TSNE est ici utilisé. Cela nous permet en effet de réduire notre nombre de dimension 
    de 100 à 2, et ainsi de pouvoir visualiser la proximité ou non des mots entre eux.
    """)
    st.image('Streamlit_App/graphs/word2vec_TSNE.jpg')


    st.markdown("""
    L'exemple ci-dessous permet de vérifier la cohérence des prédictions du modèle. \n
    Quelles sont les mots les plus similaires à notre exemple 'superleagueout' ?
    Il est possible d'essayer pour d'autres mots présent dans notre vocabulaire.
    """)
    model = Word2Vec.load("models/word2vec.model")

    col1, col2 = st.beta_columns((3,3))
    with col1:
        word_input = st.text_input('Enter some text', value='superleagueout')
    with col2:
        st.write(pd.DataFrame(model.wv.most_similar(f'{word_input}', topn=10), columns = ['Mots', 'Similarity']))
    
    