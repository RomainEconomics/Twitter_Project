from numpy import vectorize
from numpy.core import fromnumeric
from numpy.core.records import format_parser, fromarrays
import streamlit as st
import csv

#from wordcloud import WordCloud
#from unidecode import unidecode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#import re
import random

# Les bigrammes
#from collections import Counter
#from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

# Librairie pour stocker les différents modèles 
import joblib

def app():

    
    
    def display_topics(model, feature_names, no_top_words):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
        return pd.DataFrame(topic_dict)
    
    
    st.header("Saisie des données par l'utilisateur")
    # LAYING OUT THE TOP SECTION OF THE APP
    row1_1, row1_2, row1_3 = st.beta_columns((2,2,2))
    
    with row1_1:
        selected_model = st.selectbox('Modèles',['LDA', 'BigramLDA', 'NMF_Frobenius', 'NMF_K-L'])
    with row1_2:    
        selected_topics = st.selectbox('Topics',list(range(2,11,1)))
    with row1_3:
        selected_num_words = st.selectbox('Nombre de Mots',list(range(5,21,1)), index = 10)
    
    
    # Import feature names
    if selected_model == 'LDA':
        with open('models/tf_feature_names', 'r') as file:
            feature_names = list(csv.reader(file))
    elif selected_model == 'BigramLDA':
        with open('models/bigram_tf_feature_names', 'r') as file:
            feature_names = list(csv.reader(file))
    else:
        with open('models/tf_feature_names_nmf', 'r') as file:
            feature_names = list(csv.reader(file))
            
    
    # Import model scores 
    if selected_model == 'LDA':
        scores = pd.read_csv('models/LDA_scores')
    elif selected_model == 'BigramLDA':
        scores = pd.read_csv('models/bigramLDA_scores')
    else:
        pass
    
    
    if selected_model in ['LDA', 'BigramLDA']:
        # LAYING OUT THE TOP SECTION OF THE APP
        row1_1, row1_2 = st.beta_columns((3,3))
    
        with row1_1:
            st.markdown("""
            Pour le modèle LDA, deux statistiques permettent de mesurer la qualité du modèle :
            * **Score :** correspond à la log-likehood, le plus haut score suggère un meilleur modèle
            * **Perplexity :**  mesure la probabilité pour le modèle de prédire un échantillon. 
            Ici on veut sélectionner le modèle ayant un plus faible score.
            """)
    
        with row1_2:
            st.dataframe(scores)
    else:
        row1_1, row1_2 = st.beta_columns((3,3))
    
        with row1_1:
            st.markdown("""
            **Non-negative matrix factorization (NMF)** est un algorithme non supervisé.\n
            Celui-ci est utilisé dans beaucoup d'applications différentes et peut être utilisé pour 
            la réduction de dimension (à la manière du PCA).
            Une condition nécessaire est que toutes les valeurs de la matrice donnée comme input soient positives.\n
            Cet algorithme prend une matrice comme input, ici une matrice avec les Tf-Idf associés à chaque tweet,
            et va décomposer cette matrice en deux nouvelles matrices. 
            """)

        with row1_2:
            st.image('Streamlit_App/apps/NMF.png', caption="Une représentation simplifiée de l'algorithme NMF")
    
    
    # Import models
    @st.cache
    def load_data(selected_model, num_topics):
        model = joblib.load(f'models/{selected_model}_{num_topics}_topics')
    
        return model
    
    model_chosen = load_data(selected_model, selected_topics)
    
    df = display_topics(model_chosen, feature_names[0], no_top_words = (selected_num_words ))
    
    
    #st.header('Display Topics')
    #st.dataframe(df, 1000, 800)
    
    def plot_top_words(model, feature_names, n_top_words, title):
        
        if len(model.components_) > 5:
            fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
            axes = axes.flatten()
            for topic_idx, topic in enumerate(model.components_):
                top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                weights = topic[top_features_ind]
        
                ax = axes[topic_idx]
                ax.barh(top_features, weights, height=0.7, color='#3e6096')
                ax.set_title(f'Topic {topic_idx +1}',
                             fontdict={'fontsize': 30})
                ax.invert_yaxis()
                ax.tick_params(axis='both', which='major', labelsize=20)
                for i in 'top right left'.split():
                    ax.spines[i].set_visible(False)
            
                fig.suptitle(title, fontsize=40)
        else:
            fig, axes = plt.subplots(1, len(model.components_), figsize=(15, 7), sharex=True)
            axes = axes.flatten()
            for topic_idx, topic in enumerate(model.components_):
                top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                weights = topic[top_features_ind]
        
                ax = axes[topic_idx]
                ax.barh(top_features, weights, height=0.7, color='#3e6096')
                ax.set_title(f'Topic {topic_idx +1}',
                             fontdict={'fontsize': 13})
                ax.invert_yaxis()
                ax.tick_params(axis='both', which='major', labelsize=12)
                for i in 'top right left'.split():
                    ax.spines[i].set_visible(False)
            
                fig.suptitle(title, fontsize=20)
        
    
        plt.subplots_adjust(top=0.88, bottom=0.05, wspace=0.90, hspace=0.3)
    
        return fig
        
        
    
    st.pyplot(
        plot_top_words(model_chosen, feature_names[0], selected_num_words, f'Topics obtenus à partir du {selected_model} modèle')
    )
    
    
    # Import test set
    X_hold = pd.read_csv('Streamlit_App/generate_random_tweets.csv')
    
    def classify_a_tweet(model, X):
        # Import feature names
        if selected_model == 'LDA':
            # Load vectorizer, to transform the test set
            vectorizer = joblib.load('models/LDA_vectorizer')
        elif selected_model == 'BigramLDA':
            # Load vectorizer, to transform the test set
            vectorizer = joblib.load('models/BigramLDA_vectorizer')
        else:
            # Load vectorizer, to transform the test set
            vectorizer = joblib.load('models/NMF_vectorizer')
        
        weights_test_set = model.transform(vectorizer.transform(X['tweet_clean']))
        
        colnames = ["Topic" + str(i) for i in range(model.n_components)]
        docnames = ["Tweet" + str(i) for i in range(len(X))]
        df_doc_topic = pd.DataFrame(np.round(weights_test_set, 2), columns=colnames, index=docnames)
        significant_topic = np.argmax(df_doc_topic.values, axis=1)
        df_doc_topic['Topic Dominant'] = significant_topic
        
        idx = random.sample(range(len(X)), 1)[0]
        tweet = X.iloc[idx, [3,4]]
        
        return tweet, df_doc_topic.iloc[idx]
    
    
    if st.button('Classify a tweet'):
    
        t, df1 = classify_a_tweet(model_chosen, X_hold)
        st.text("Tweet original:")
        st.write(t[0])
        st.text("Tweet nettoyé:")
        st.write(t[1])
        st.dataframe(pd.DataFrame(df1).T)