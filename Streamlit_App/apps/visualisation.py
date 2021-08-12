import streamlit as st

def app():
    st.title('Visualisation')

    #st.sidebar.markdown("## Select Data Time and Detector")

    select_event = st.selectbox('Choisissez un graphique',
                                    ['Nombre de de tweets par jour', 'Hashtags', 
                                    'Unigram', 'Bigram',
                                    'Personnalités', 'Clubs'])


    if select_event == 'Nombre de de tweets par jour':
        st.image("Streamlit_App/graphs/tweets_par_jour.png")

    elif select_event == 'Hashtags':
        st.image("Streamlit_App/graphs/hashtags.jpg")

    elif select_event == 'Unigram':
        st.image("Streamlit_App/graphs/unigram.png")

    elif select_event == 'Bigram':
        st.image("Streamlit_App/graphs/bigram.png")

    elif select_event == 'Personnalités':
        st.image("Streamlit_App/graphs/personnalites.png")

    elif select_event == 'Clubs':
        st.image("Streamlit_App/graphs/clubs.png")
    else:
        pass