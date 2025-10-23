import streamlit as st
import pandas as pd
import numpy as np
import math
import altair as alt


st.title('Projet oiseaux :bird: de :blue[Marseille] :sunglasses:')
st.text("Léa COQUEREAU\nGuillaume VALENTIN\nAndreas JULIEN-CARAGUEL\n")
st.header("Problématique", divider="gray")
st.link_button("Source de la donnée", "https://www.data.gouv.fr/fr/datasets/marseille-biodiversite-oiseaux/")
st.text("Comment les différentes espèces d'oiseaux de Marseille sont-elles réparti dans la ville ?")
