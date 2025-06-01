# -*- coding: utf-8 -*-
"""
Created on Fri May 30 19:55:29 2025

@author: tere8
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# 1. Cargar el CSV
df = pd.read_csv("playa_tmax_wspd.csv", parse_dates=['fecha'])

# 2. Variables
X = df[['tmax', 'wspd']].values
y = df['aptoplay'].astype(int).values

# 3. División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Valores de R
Rs = [0.1, 1.0, 10.0, 1e6]

# 5. Colormap personalizado: fondo y puntos
cmap_light = ListedColormap(["#BBDDFF", "#BBFFBB"])   # fondo: azul claro (-1), verde claro (+1)
cmap_bold  = ["blue", "#66CC66"]                      # puntos: azul, verde clarito

# 6. Entrenamiento y visualización
for R in Rs:
    clf = LinearSVC(C=R, loss='squared_hinge', dual=False, max_iter=10000, random_state=42)
    clf.fit(X_train, y_train)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    # Rejilla
    x_min, x_max = 0, 45
    y_min, y_max = 0, 20
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Hiperplano y márgenes
    x_vals = np.linspace(x_min, x_max, 500)
    y_hyper = -(w[0]*x_vals + b) / w[1]
    y_margin_pos = -(w[0]*x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0]*x_vals + b + 1) / w[1]

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Fondo con regiones coloreadas
    ax.contourf(xx, yy, Z > 0, alpha=0.3, cmap=cmap_light)

    # Puntos según clase
    X_pos = X[y == 1]
    X_neg = X[y == -1]
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color=cmap_bold[1], edgecolor='k', label='Día apto (+1)')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color=cmap_bold[0], edgecolor='k', label='Día no apto (-1)')

    # Hiperplano y márgenes
    ax.plot(x_vals, y_hyper, 'k-', label=r'$D(\vec{x}) = 0$ (hiperplano)')
    ax.plot(x_vals, y_margin_pos, 'k--', label=r'$D(\vec{x}) = +1$ (margen)')
    ax.plot(x_vals, y_margin_neg, 'k--', label=r'$D(\vec{x}) = -1$ (margen)')

    # Estética
    ax.set_xlim(12, 43)
    ax.set_ylim(0, 14)
    ax.set_xlabel("tmax (°C)")
    ax.set_ylabel("wspd (km/h)")
    ax.set_title(f"Clasificación SVM con R = {R}")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
