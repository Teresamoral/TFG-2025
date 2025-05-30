# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:08:15 2025

@author: tere8
"""

# Código completo para entrenar y visualizar un árbol de decisión con profundidad 2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Cargar los datos
df = pd.read_csv('playa_tmax.csv')

# Definir variables predictoras y etiqueta
X = df[['tmax']]
y = df['aptoplay']

# División en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar árbol de decisión con profundidad máxima 2
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

# Evaluar precisión
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Visualizar el árbol con cortes y precisión
plt.figure(figsize=(12, 6))
plot_tree(
    clf,
    feature_names=['tmax'],
    class_names=['-1', '+1'],
    filled=True,
    rounded=True,
    precision=2
)
plt.title(f"Árbol de decisión (profundidad máxima 2) - Precisión: {accuracy:.3f}")
plt.show()
