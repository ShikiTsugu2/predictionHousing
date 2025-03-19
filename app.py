import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les modèles
best_model_regression = joblib.load("best_model_regression.joblib")
best_model_classification = joblib.load("best_model_classification.joblib")

# Interface utilisateur Streamlit
st.title("Prédiction du Prix d'une Maison ou de sa Catégorie")

# Choix du type de prédiction
type_prediction = st.selectbox("Choisissez le type de prédiction :", ["Régression", "Classification"])

# Définir les features en fonction du type de prédiction
if type_prediction == "Régression":
    st.subheader("Prédiction du Prix des Maisons")
    feature_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
                     'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                     'parking', 'prefarea', 'furnishingstatus']
    model = best_model_regression
    label = "Prix estimé"
else:
    st.subheader("Classification : Maison chère ou non")
    feature_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
                     'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                     'parking', 'prefarea', 'furnishingstatus']
    model = best_model_classification
    label = "Catégorie estimée (1 : Maison chère / 0 : Maison abordable)"

# Entrée des données
input_data = {}
for col in feature_names:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Convertir en DataFrame
input_df = pd.DataFrame([input_data])

# Bouton de prédiction
if st.button("Prédire"):
    prediction = model.predict(input_df)
    st.write(f"### {label}: {prediction[0]}")

# Affichage de distributions
st.subheader("Répartition des prix des maisons")
data = pd.read_csv('cleaned_housing.csv')
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data['price'], kde=True, bins=30, ax=ax)
st.pyplot(fig)

# Heatmap de corrélation
st.subheader("Corrélation entre les variables")
numeric_data = data.select_dtypes(include=['float64', 'int64'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
