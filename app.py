import streamlit as st
import pandas as pd
import numpy as np
import math
import altair as alt


st.title('Projet :clapper: :red[CineStat] :clapper:')
st.text("Lise AYMONIN\nAndreas JULIEN-CARAGUEL\n")
st.header("Problématique", divider="gray")
st.link_button("Source de la donnée", "https://www.data.gouv.fr/datasets/frequentation-des-salles-de-cinema/")
st.text("A quoi ressemblera la courbe des entrées pour les mois, années à venir ?\nQuelles sont les semaines et les mois de plus forte affluence dans l’année ?")


fichier = 'Mise_en_forme_Frequentation_Salles_Cine.xlsx'
data = pd.read_excel(fichier, sheet_name ='Entrees_mois')


st.header("Tableau de données (brut)", divider=True)
data.insert(0, 'ID', range(1, 1 + len(data)))
df = pd.DataFrame(data)
df
