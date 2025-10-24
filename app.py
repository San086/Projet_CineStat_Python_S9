import streamlit as st
import pandas as pd
import numpy as np
import math
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib.patches import Patch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



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
st.header("🎬 Prévision du nombre d’entrées cinéma", divider=True)

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

# --- Tableau restructuré pour modélisation ---
st.header("Tableau des données restructurées")
df_long.insert(0, 'ID', range(1, 1 + len(df_long)))
st.dataframe(df_long)

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
splitDate = st.input(
    "Entrer une date à partir de laquelle vérifier le modèle :",
    value = "2017,01,01"
)
splitDate = pd.to_datetime(splitDate)
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
# --- Affichage des métriques ---
st.subheader("📊 Évaluation du modèle Random Forest")
st.write(f"**MAE :** {mae:,.0f} entrées")
if mae > 2000000: st.write("Erreur supérieure à 2 millions d'entrées : erreur élevée !!")
else: st.write("Erreur inférieur à 2 millions d'entrées : c'est acceptable.")

# --- Graphique matplotlib ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(datesTest, ytest, label="Valeurs réelles", color="blue", marker='o')
ax.plot(datesTest, ypred, label="Prédictions", color="orange", marker='x', linewidth=2)
ax.set_title(f"Prédiction de la fréquentation des cinémas depuis {splitDate[0:4]}")
ax.set_xlabel("Année")
ax.set_ylabel("Nombre d'entrées en millions")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# --- Affichage dans Streamlit ---
st.pyplot(fig)

# --- Commentaire ---
st.write(
    "Ce graphique compare les prévisions et les valeurs réelles du le nombre d’entrées"
    "dans les salles de cinéma françaises, basé sur un modèle de **Random Forest**. "
    "On va voir que les pics récurent correspondent aux périodes estivales et de fêtes de fin d’année."
)





# --- Titre ---
st.header("🎬 Projection du nombre d’entrées cinéma à l'avenir", divider=True)

# --- Modèle Random Forest ---
col = ['Années', 'numéro mois', 'vacances', 'lag1', 'lag2', 'mean3', 'trimestre']
x = df_long[col]
y = df_long["entrees"]

model = RandomForestRegressor(random_state=42, n_estimators=300)
model.fit(x, y)

# --- Création du futur (2025 → 2030) ---
endDate = "2030-12-01"
futur = pd.date_range(start=df_long["date"].max() + pd.offsets.MonthBegin(1), end=endDate, freq="MS")
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
ax.plot(fut_pred["date"], fut_pred["entrees"], label=f"Prévisions futures (2025–{endDate[0:4]})", color="green", linewidth=2)
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
    f"Le modèle Random Forest prédit l’évolution du nombre d’entrées cinéma jusqu’en **{futur[-1].year}**. "
    "Les valeurs réelles (en bleu) et les projections (en vert) permettent d’anticiper les tendances "
    "saisonnières et les périodes de forte affluence. Cependant les crises majeures impactant le milieu ne sont pas prises en compte (covid, évolutions technologiques, etc)."
)




# --- Titre ---
st.header("🎬 Analyse KNN — Classification des mois selon leur affluence moyenne", divider=True)

# --- Lecture du fichier Excel ---
fichier = "Mise_en_forme_Frequentation_Salles_Cine.xlsx"
df = pd.read_excel(fichier, sheet_name="Entrees_mois")

# --- Liste des mois ---
mois = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre"
]

# --- Calcul des moyennes mensuelles ---
moyennes_mois = df[mois].mean()

# --- Calcul des quantiles ---
q1 = moyennes_mois.quantile(0.33)
q2 = moyennes_mois.quantile(0.66)

# --- Attribution des catégories ---
labels = []
for val in moyennes_mois:
    if val <= q1:
        labels.append("faible affluence")
    elif val <= q2:
        labels.append("moyenne affluence")
    else:
        labels.append("forte affluence")

# --- Préparation des données ---
X = moyennes_mois.values.reshape(-1, 1)
y = labels

# --- Modèle KNN ---
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X, y)
y_pred = classifier.predict(X)

# --- Évaluation du modèle ---
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred, output_dict=True)
accuracy = accuracy_score(y, y_pred)

# --- Affichage des métriques ---
st.subheader("📊 Évaluation du modèle KNN")
st.write(f"**Exactitude du modèle :** {accuracy:.2f}")
st.write("**Matrice de confusion :**")
st.dataframe(pd.DataFrame(conf_matrix, index=sorted(set(y)), columns=sorted(set(y))))
st.write("**Rapport de classification :**")
st.dataframe(pd.DataFrame(class_report).transpose())

# --- Couleurs selon catégorie ---
palette = {"faible affluence": "lightblue", "moyenne affluence": "orange", "forte affluence": "red"}

# --- Graphique ---
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(x=moyennes_mois.index, y=moyennes_mois.values, palette=[palette[c] for c in y_pred], ax=ax)
ax.set_title("Classification des mois selon leur affluence moyenne (KNN)")
ax.set_ylabel("Entrées moyennes")
ax.set_xlabel("Mois")
plt.xticks(rotation=45)

# --- Légende dynamique ---
legend_elements = [
    Patch(facecolor="lightblue", label=f"Faible affluence ≤ {int(q1):,} entrées"),
    Patch(facecolor="orange", label=f"Moyenne affluence {int(q1)+1:,} – {int(q2):,} entrées"),
    Patch(facecolor="red", label=f"Forte affluence > {int(q2):,} entrées"),
]
ax.legend(handles=legend_elements, title="Catégorie d'affluence")

st.pyplot(fig)

# --- Interprétation ---
st.subheader("🧩 Interprétation")
st.write(
    "Cette classification utilise un algorithme **K-Nearest Neighbors (KNN)** "
    "pour identifier les mois de **forte**, **moyenne** ou **faible affluence** "
    "en fonction du nombre moyen d’entrées au cinéma."
)
st.info(
    "👉 En rouge : forte affluence (été et fêtes), "
    "en orange : moyenne affluence, "
    "en bleu : faible affluence."
)

st.header("Surprises", divider=True)
st.link_button("Clique pour une surprise", "https://www.youtube.com/watch?v=xvFZjo5PgG0")
st.link_button("Clique pour une surprise v2", "https://chat-jai-pete.fr/")
