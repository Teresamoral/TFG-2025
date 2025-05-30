# -*- coding: utf-8 -*-
"""
Created on Tue May 27 22:06:34 2025

@author: tere8
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# 1) Carga el dataset
df = pd.read_csv(r"C:\Users\tere8\Desktop\UNIVERSIDAD\CUARTO\MODELIZACIÓN\playa_dataset_presMax.csv")

# 2) Asegura tipos numéricos y elimina filas con NaN
for col in ['tmax', 'wspd', 'presMax', 'aptoplay']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['tmax', 'wspd', 'presMax', 'aptoplay'])

# 3) Separa características y etiqueta
X = df[['tmax', 'wspd', 'presMax']].values
y = df['aptoplay'].values.astype(int)

# 4) Train/test split estratificado 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5) Entrena el árbol de decisión
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 6) Extrae reglas y precisiones
rules = export_text(clf, feature_names=['tmax','wspd','presMax'])
acc_train = clf.score(X_train, y_train)
acc_test  = clf.score(X_test, y_test)

# 7) Muestra resultados
print("Decision Tree rules:")
print(rules)
print(f"Train accuracy: {acc_train:.4f}")
print(f"Test  accuracy: {acc_test:.4f}")
