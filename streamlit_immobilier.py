# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:57:06 2023

@author: PC.054
"""

import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import r2_score

df=pd.read_csv("Housing.csv")

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write("Ce projet s'inscrit dans un contexte immobilier. L'objectif est de prédire le prix d'un logement à partir de ses caractéristiques, dans un cadre d'estimations financières.")
    
    st.write("Nous avons à notre disposition le fichier housing.csv qui contient des données immobilières. Chaque observation en ligne correspond à un logement. Chaque variable en colonne est une caractéristique de logement.")
    
    st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire le prix.")
    
    st.image("immobilier.jpg")
    
elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
        
elif page == pages[2]:
    st.write("### Analyse de données")
    
    fig = sns.displot(x='price', data=df, kde=True)
    plt.title("Distribution de la variable cible price")
    st.pyplot(fig)
    
    fig2 = px.scatter(df, x="price", y="area", title="Evolution du prix en fonction de la surface")
    st.plotly_chart(fig2)
    
    fig3, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    plt.title("Matrice de corrélation des variables du dataframe")
    st.write(fig3)

elif page == pages[3]:
    st.write("### Modélisation")
    
    df_prep = pd.read_csv("df_preprocessed.csv")
    
    y = df_prep["price"]
    X= df_prep.drop("price", axis=1)
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler()
    num = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    X_train[num] = scaler.fit_transform(X_train[num])
    X_test[num] = scaler.transform(X_test[num])
    
    reg = joblib.load("model_reg_line")
    rf = joblib.load("model_reg_rf")
    knn = joblib.load("model_reg_knn")
    
    y_pred_reg=reg.predict(X_test)
    y_pred_rf=rf.predict(X_test)
    y_pred_knn=knn.predict(X_test)
    
    model_choisi = st.selectbox(label = "Modèle", options = ['Regression Linéaire', 'Random Forest', 'KNN'])
    
    def train_model(model_choisi) : 
        if model_choisi == 'Regression Linéaire' :
            y_pred = y_pred_reg
        elif model_choisi == 'Random Forest' :
            y_pred = y_pred_rf
        elif model_choisi == 'KNN' :
            y_pred = y_pred_knn
        r2 = r2_score(y_test, y_pred)
        return r2
    
    st.write("Coefficient de détermination", train_model(model_choisi))