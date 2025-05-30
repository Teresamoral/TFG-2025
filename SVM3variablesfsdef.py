# -*- coding: utf-8 -*-
"""
Created on Wed May 28 19:57:10 2025

@author: tere8
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv("playa_dataset_ext_sol.csv")

# Seleccionar variables
X = df[["tmax", "wspd", "sol"]].values
y = df["aptoplay"].astype(int).values

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Entrenar SVM de margen blando
svm = LinearSVC(C=1.0, loss='squared_hinge', dual=False, max_iter=10000, random_state=42)
svm.fit(X_train, y_train)

# Extraer coeficientes del hiperplano
w_tmax, w_wspd, w_sol = svm.coef_[0]
b = svm.intercept_[0]

# Calcular precisión
train_acc = accuracy_score(y_train, svm.predict(X_train))
test_acc = accuracy_score(y_test, svm.predict(X_test))

# Mostrar resultados
print(f"Hiperplano: f(x) = {w_tmax:.4f}·tmax + {w_wspd:.4f}·wspd + {w_sol:.4f}·sol + ({b:.4f}) = 0")
print(f"Accuracy en train: {train_acc:.4f}")
print(f"Accuracy en test : {test_acc:.4f}")
