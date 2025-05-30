# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:22:36 2025

@author: tere8
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv('playa_dataset_ext_sol.csv')  # Asegúrate de que el archivo esté en el mismo directorio

# Definir variables predictoras y etiqueta
X = df[['tmax', 'wspd', 'sol']]
y = df['aptoplay']

# Dividir en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el árbol de decisión con profundidad máxima 4
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Calcular precisión en test y entrenamiento
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Imprimir precisión
print(f"Precisión en entrenamiento: {accuracy_train:.3f}")
print(f"Precisión en test: {accuracy_test:.3f}")

# Visualizar el árbol
plt.figure(figsize=(16, 8))
plot_tree(
    clf,
    feature_names=['tmax', 'wspd', 'sol'],
    class_names=['-1', '+1'],
    filled=True,
    rounded=True,
    precision=2
)
plt.title(f"Árbol de decisión (tmax, wspd, sol) - Precisión test: {accuracy_test:.3f}")
plt.show()
