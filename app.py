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





# --- Titre ---
st.title("🎬 CineStat — Projection du nombre d’entrées cinéma jusqu’en 2030")

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

# --- Création de la colonne date ---
df_long["date"] = pd.to_datetime(dict(year=df_long["Années"], month=df_long["numéro mois"], day=1))
df_long = df_long.sort_values("date").reset_index(drop=True)
df_long["trimestre"] = df_long["date"].dt.quarter
df_long["vacances"] = df_long["numéro mois"].isin([7, 8, 12]).astype(int)

# --- Définition des lags ---
s1, s2, m = 3, 6, 12
df_long["lag1"] = df_long["entrees"].shift(s1)
df_long["lag2"] = df_long["entrees"].shift(s2)
df_long["mean3"] = df_long["entrees"].rolling(m).mean()
df_long = df_long.dropna().reset_index(drop=True)

# --- Modèle Random Forest ---
col = ['Années', 'numéro mois', 'vacances', 'lag1', 'lag2', 'mean3', 'trimestre']
x = df_long[col]
y = df_long["entrees"]

model = RandomForestRegressor(random_state=42, n_estimators=300)
model.fit(x, y)

# --- Création du futur (2025 → 2030) ---
futur = pd.date_range(start=df_long["date"].max() + pd.offsets.MonthBegin(1), end="2030-12-01", freq="MS")
dfutur = pd.DataFrame({"date": futur})
dfutur["Années"] = dfutur["date"].dt.year
dfutur["numéro mois"] = dfutur["date"].dt.month
dfutur["trimestre"] = dfutur["date"].dt.quarter
dfutur["vacances"] = dfutur["numéro mois"].isin([7, 8, 12]).astype(int)

# --- Concaténation ---
dfull = pd.concat([df_long, dfutur], ignore_index=True)

# --- Génération des prédictions futures ---
for i in range(len(df_long), len(dfull)):
    dfull.loc[i, "lag1"] = dfull.loc[i - s1, "entrees"]
    dfull.loc[i, "lag2"] = dfull.loc[i - s2, "entrees"]
    dfull.loc[i, "mean3"] = dfull.loc[i - m:i - 1, "entrees"].mean()
    xfut = dfull.loc[i, col].to_frame().T
    dfull.loc[i, "entrees"] = model.predict(xfut)[0]

# --- Extraction des prévisions futures ---
fut_pred = dfull.loc[len(df_long):, ["date", "entrees"]]

# --- Graphique ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_long["date"], df_long["entrees"], label="Valeurs réelles", color="blue")
ax.plot(fut_pred["date"], fut_pred["entrees"], label="Prévisions futures (2025–2030)", color="green", linewidth=2)
ax.set_title(f"Prévision du nombre d’entrées cinéma jusqu’en {futur[-1].year}")
ax.set_xlabel("Date")
ax.set_ylabel("Nombre d’entrées (en millions)")
ax.legend()
ax.grid(True)
plt.tight_layout()

# --- Affichage du graphique ---
st.pyplot(fig)

# --- Texte d’analyse ---
st.subheader("📈 Interprétation")
st.write(
    f"Le modèle Random Forest prédit l’évolution du nombre d’entrées cinéma jusqu’à **{futur[-1].year}**. "
    "Les valeurs réelles (en bleu) et les projections (en vert) permettent d’anticiper les tendances "
    "saisonnières et les périodes de forte affluence."
)
