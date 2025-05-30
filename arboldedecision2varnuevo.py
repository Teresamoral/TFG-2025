# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:29:54 2025

@author: tere8
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# 1) Ruta al CSV generado con tmax, wspd y aptoplay
csv_path = r"C:\Users\tere8\Desktop\UNIVERSIDAD\CUARTO\MODELIZACIÓN\playa_tmax_wspd.csv"

# 2) Cargo los datos
df = pd.read_csv(csv_path, parse_dates=["fecha"])
X = df[["tmax", "wspd"]].values
y = df["aptoplay"].astype(int).values

# 3) Divido en train (70%) / test (30%) con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 4) Entreno el árbol de decisión
tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,       # por ejemplo, profundidad máxima 2
    random_state=42
)
tree.fit(X_train, y_train)

# 5) Métricas de rendimiento
acc_train = tree.score(X_train, y_train)
acc_test  = tree.score(X_test,  y_test)
print(f"Accuracy train: {acc_train:.4f}")
print(f"Accuracy test : {acc_test:.4f}")

# 6) Reglas textuales
rules = export_text(tree, feature_names=["tmax","wspd"])
print("\nReglas del árbol:\n")
print(rules)

# 7) Dibujo del árbol
plt.figure(figsize=(8,6))
plot_tree(
    tree,
    feature_names=["tmax","wspd"],
    class_names=["-1","+1"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árbol de Decisión (tmax, wspd)")
plt.show()
