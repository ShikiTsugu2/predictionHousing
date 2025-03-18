import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le modèle
best_model_pipeline = joblib.load("best_model_pipeline.joblib")

# Interface utilisateur Streamlit
st.title("Prédiction du Prix des Maisons")
st.write("Entrez les caractéristiques de la maison pour obtenir une estimation du prix.")

# Définir les colonnes d'entrée (ajuste selon ton dataset)
feature_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus']
input_data = {}

for col in feature_names:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Convertir en DataFrame
input_df = pd.DataFrame([input_data])

# Bouton de prédiction
def predict_price():
    prediction = best_model_pipeline.predict(input_df)
    return prediction[0]

if st.button("Prédire"):
    result = predict_price()
    st.write(f"### Prix estimé: {result:,.0f} $")

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
