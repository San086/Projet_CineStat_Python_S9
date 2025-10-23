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
