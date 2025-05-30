# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:43:11 2025

@author: tere8
"""

""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# 1) Cargar el CSV
df = pd.read_csv('playa_dataset_presMax.csv', parse_dates=['fecha'])
print("Total observaciones:", len(df))

# 2) Definir variables
X = df[['tmax', 'wspd', 'presMax']].values
y = df['aptoplay'].astype(int).values

# Eliminar cualquier fila con valores NaN o inf
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]
print(f"Filas válidas tras eliminar NaN/infs: {X.shape[0]}")

# 3) Dividir en train (70%) y test (30%), estratificando según y
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
print("Observaciones en train:", X_train.shape[0])
print("Observaciones en test :", X_test.shape[0])

# 4) Entrenar SVM de margen blando en primal
R = 1.0
svm = LinearSVC(
    C=R,
    loss='squared_hinge',
    dual=False,
    max_iter=10000,
    random_state=42
)
svm.fit(X_train, y_train)

# 5) Extraer hiperplano óptimo (w, b)
w = svm.coef_[0]
b = svm.intercept_[0]
print("\nHiperplano óptimo:")
print(f"   {w[0]:.4f}·tmax + {w[1]:.4f}·wspd + ({w[2]:.4f})·presMax + ({b:.4f}) = 0")

# 6) Ancho efectivo del margen
margin_width = 2.0 / np.linalg.norm(w)
print(f"\nAncho de margen (2/||w||): {margin_width:.4f}")

# 7) Cálculo de slacks ξᵢ en entrenamiento
decision = svm.decision_function(X_train)
xi = np.maximum(0, 1 - y_train * decision)
print(f"\nSuma de slacks Σξᵢ = {xi.sum():.4f} → penalización total R·Σξ = {R * xi.sum():.4f}")

# 8) Aproximación de vectores soporte
sv_idx = np.where((xi > 1e-3) | (np.abs(1 - y_train * decision) < 1e-2))[0]
print(f"\nNº aproximado de vectores soporte: {len(sv_idx)}")
print("Algunos vectores soporte (índice, tmax, wspd, presMax, etiqueta, f(x)):")
for i in sv_idx[:5]:
    print(f"  [{i}] = {X_train[i]}, y = {y_train[i]}, f(x) = {decision[i]:.3f}")

# 9) Accuracy en train y test
print(f"\nAccuracy train: {svm.score(X_train, y_train):.3f}")
print(f"Accuracy test : {svm.score(X_test,  y_test):.3f}")
