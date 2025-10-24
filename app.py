import streamlit as st
import pandas as pd
import numpy as np
import math
import altair as alt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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



# --- Titre ---
st.title("🎬 Projet CineStat — Prévision du nombre d’entrées cinéma")

# --- Chargement du fichier Excel ---
fichier = 'Mise_en_forme_Frequentation_Salles_Cine.xlsx'
data = pd.read_excel(fichier, sheet_name='Entrees_mois')

# --- Restructuration du jeu de données ---
df_long = data.melt(
    id_vars=["Années"],
    value_vars=data.columns[2:13],
    var_name="mois", value_name="entrees"
)

# --- Mois en chiffres ---
mois_map = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
}
df_long["numéro mois"] = df_long["mois"].map(mois_map)

# --- Date complète ---
df_long["date"] = pd.to_datetime(dict(year=df_long["Années"], month=df_long["numéro mois"], day=1))
df_long = df_long.sort_values("date").reset_index(drop=True)
df_long["trimestre"] = df_long["date"].dt.quarter
df_long["vacances"] = df_long["numéro mois"].isin([7, 8, 12]).astype(int)

# --- Création de features de décalage ---
s1, s2, m = 3, 6, 12
df_long["lag1"] = df_long["entrees"].shift(s1)
df_long["lag2"] = df_long["entrees"].shift(s2)
df_long["mean3"] = df_long["entrees"].rolling(m).mean()

# --- Suppression des NaN créés par les décalages ---
df_long = df_long.dropna().reset_index(drop=True)

# --- Variables explicatives et cible ---
col = ['Années', 'numéro mois', 'vacances', 'lag1', 'lag2', 'mean3', 'trimestre']
x = df_long[col]
y = df_long["entrees"]

# --- Séparation train/test ---
splitDate = "2017-01-01"
xtrain = x[df_long["date"] < splitDate]
ytrain = y[df_long["date"] < splitDate]
xtest = x[df_long["date"] >= splitDate]
ytest = y[df_long["date"] >= splitDate]
datesTest = df_long.loc[df_long["date"] >= splitDate, "date"]

# --- Entraînement du modèle Random Forest ---
model = RandomForestRegressor(random_state=42, n_estimators=300)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

# --- Évaluation du modèle ---
mae = mean_absolute_error(ytest, ypred)
mape = np.mean(np.abs((ytest - ypred) / (ytest + 0.1))) * 100

# --- Affichage des métriques ---
st.subheader("📊 Évaluation du modèle Random Forest")
st.write(f"**MAE :** {mae:,.0f} entrées")
st.write(f"**Erreur moyenne (MAPE) :** {mape:.2f}%")

# --- Graphique matplotlib ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_long["date"], df_long["entrees"], label="Valeurs réelles", color="blue")
ax.plot(datesTest, ypred, label="Prédictions", color="orange", linewidth=2)
ax.set_title("Prévision du nombre d’entrées cinéma (modèle Random Forest)")
ax.set_xlabel("Année")
ax.set_ylabel("Nombre d'entrées en millions")
ax.legend()
ax.grid(True)
plt.tight_layout()

# --- Affichage dans Streamlit ---
st.pyplot(fig)

# --- Commentaire ---
st.write(
    "Ce graphique illustre les valeurs observées et les prévisions du nombre d’entrées "
    "dans les salles de cinéma françaises, basées sur un modèle de **Random Forest**. "
    "Les pics observés correspondent aux périodes estivales et de fêtes de fin d’année."
)
