# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:50:18 2025

@author: tere8
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv("playa_tmax.csv")

# Preparar variables
X = df[["tmax"]].values
y = df["aptoplay"].astype(int).values

# Separar en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Entrenar SVM de margen blando
svm = LinearSVC(C=1.0, loss='squared_hinge', dual=False, max_iter=10000, random_state=42)
svm.fit(X_train, y_train)

# Extraer coeficientes del hiperplano
w = svm.coef_[0][0]
b = svm.intercept_[0]

# Calcular precisión
train_acc = accuracy_score(y_train, svm.predict(X_train))
test_acc = accuracy_score(y_test, svm.predict(X_test))

# Mostrar hiperplano y precisión
print(f"Hiperplano: f(x) = {w:.3f}·tmax + ({b:.3f})")
print(f"Accuracy en train: {train_acc:.3%}")
print(f"Accuracy en test : {test_acc:.3%}")
