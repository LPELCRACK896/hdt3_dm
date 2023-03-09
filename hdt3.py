#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:49:22 2023

@author: Guillermo Furlan, Luis Pedro Gonzalez 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC


pd.options.mode.chained_assignment = None
#.data
datos = pd.read_csv('./data/wine_fraud.csv')

def analisis_exploratorio():
    datos = pd.read_csv('./data/wine_fraud.csv')
    #Analisis exploratorio

    #Estadpistiica descriptiva
    profile = ProfileReport(datos, title="Pandas Profiling Report")
    profile.to_file("report.html")
    #Otros graficos
    quantitative = list(datos.columns)[:-2]
    datos[quantitative].boxplot()
    plt.show()

def svm():
    datos = pd.read_csv('./data/wine_fraud.csv')
    #analisis_exploratorio()
    #Limpieza
    datos=datos.drop(['fixed acidity','type','citric acid','free sulfur dioxide','total sulfur dioxide','alcohol' ], axis=1)
    corr_df = datos.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True)

    #Dividir conjunto de datos
    X=datos.loc[:,datos.columns!="quality"]
    y=datos.loc[:,datos.columns=="quality"]

    #Dividir datoa de prueba y de entrenamiento
    X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.25, random_state = 0)

    #Normalizar Datos
    normalizador = StandardScaler()
    X_entreno = normalizador.fit_transform(X_entreno)
    X_prueba = normalizador.transform(X_prueba)

    clasificador = SVC(kernel = 'linear', random_state = 0)
    clasificador.fit(X_entreno, y_entreno)

    #Predecir
    y_pred = clasificador.predict(X_prueba)

    mat_conf = confusion_matrix(y_prueba, y_pred)
    print("###MATRIZ DE CONFUSION###")
    print(mat_conf)
    score = accuracy_score(y_prueba, y_pred)
    print("###Accuracy score###")
    print(score)