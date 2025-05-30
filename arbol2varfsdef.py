# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:15:37 2025

@author: tere8
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv('playa_tmax_wspd.csv')  # Asegúrate de tener el archivo en el mismo directorio

# Seleccionar variables predictoras y etiqueta
X = df[['tmax', 'wspd']]
y = df['aptoplay']

# División en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del árbol de decisión
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluación del modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.3f}")

# Visualización del árbol
plt.figure(figsize=(14, 7))
plot_tree(
    clf,
    feature_names=['tmax', 'wspd'],
    class_names=['-1', '+1'],
    filled=True,
    rounded=True,
    precision=2
)
plt.title(f"Árbol de decisión (tmax + wspd, profundidad 4) - Precisión: {accuracy:.3f}")
plt.show()
