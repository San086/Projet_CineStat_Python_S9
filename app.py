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


fichier = "marseille_biodiversite_oiseaux_parcs.csv"
data = pd.read_csv(fichier)

st.header("Tableau de données (brut)", divider=True)
data.insert(0, 'ID', range(1, 1 + len(data)))
df = pd.DataFrame(data)
df


st.header("Répartition du nombre d'individus par type de parc", divider=True)
if "Type" not in data.columns:
    st.error("La colonne 'Type' est manquante dans les données.")
    st.stop()

type_counts = data["Type"].value_counts().reset_index()
type_counts.columns = ["Type", "Count"]

chart = (
    alt.Chart(type_counts)
    .mark_bar(color="skyblue", stroke="black")
    .encode(
        x=alt.X("Type", title="Type"),
        y=alt.Y("Count", title="Nombre d'individu observé"),
        tooltip=["Type", "Count"]
    )
    .properties(
        title="Distribution du type de parc",
        width=600,
        height=400,
    )
    .configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    .configure_title(
        fontSize=16
    )
)

st.altair_chart(chart, use_container_width=True)




st.header("Tableau du nombre d'espèces", divider=True)
tab1 = data["Nom vernaculaire"].value_counts()
tab1



st.header("Nombre d'espèces observées par site", divider=True)
if "Nom du site" not in data.columns:
    st.error("La colonne 'Nom du site' est manquante dans les données.")
    st.stop()

site_counts = data["Nom du site"].value_counts().reset_index()
site_counts.columns = ["Nom du site", "Fréquence"]

chart = (
    alt.Chart(site_counts)
    .mark_bar(color="skyblue")
    .encode(
        x=alt.X("Nom du site", sort="-y", title="Nom du site"),
        y=alt.Y("Fréquence", title="Nombre d'espèces observées"),
        tooltip=["Nom du site", "Fréquence"],
    )
    .properties(
        width=800,
        height=400,
    )
    .configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    )
    .configure_title(fontSize=16)
)

st.altair_chart(chart, use_container_width=True)



st.header("Tableau récapitulatif de la répartition des espèces par site", divider=True)
arcs = pd.DataFrame({
    "Nom du site": data["Nom du site"].unique(),
    "Nombre d'espèces observées": data.groupby("Nom du site")["Nom vernaculaire"].nunique().values,
    "Type": data.groupby("Nom du site")["Type"].first().values,  # Supposons que chaque parc ait un seul type
    "Espèces observées": data.groupby("Nom du site")["Nom vernaculaire"].apply(list).values,
    "Adresse": data.groupby("Nom du site")["Adresse"].first().values,  # Supposons que chaque parc ait une adresse unique
    "Latitude": data.groupby("Nom du site")["Latitude"].first().values,
    "Longitude": data.groupby("Nom du site")["Longitude"].first().values
})

arcs


nom_vernaculaire_selection = st.selectbox(
    "Sélectionnez une espèce pour voir où elle a été observée :", 
    options=data["Nom vernaculaire"].unique(),
    index=None,
    placeholder="Selectionne une espèce d'oiseau...",
)

espece_info = data[data["Nom vernaculaire"] == nom_vernaculaire_selection]
sites_observes = espece_info["Nom du site"].unique()

st.write(f"**Sites où l'espèce '{nom_vernaculaire_selection}' a été observée :**")
st.dataframe(pd.DataFrame({"Nom du site                                                       ": sites_observes}))

st.write(f"**Nombre total de sites où cette espèce a été observée :** {len(sites_observes)}")



st.header("Carte de la localisation des parcs", divider=True)

if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
    st.error("Le fichier doit contenir les colonnes 'Latitude' et 'Longitude'.")
    st.stop()

data = data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

try:
    mappy = data[['latitude', 'longitude']].dropna() 
    st.map(mappy)
except Exception as e:
    st.error(f"Une erreur est survenue lors de la création de la carte : {e}")
    


st.header("Surprises", divider=True)
st.link_button("Clique pour une surprise", "https://www.youtube.com/watch?v=xvFZjo5PgG0")
st.link_button("Clique pour une surprise v2", "https://chat-jai-pete.fr/")
st.link_button("Clique pour une surprise (Loïc uniquement)", "https://youtube.com/clip/UgkxJYuLWF-dmcaVzJN7aGK6j6cXH4Rg4bsD?si=LVFHsB_TEh_y_Yi2")
st.link_button("Clique pour une surprise (Hamilton uniquement)", "https://youtube.com/clip/UgkxuyJ8YCM5KE6kiz7_0NDUBI9nT3pizi7V?si=iJsjVDl-V46x6v_D")
