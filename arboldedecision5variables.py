# -*- coding: utf-8 -*-
"""
Created on Tue May 27 21:28:47 2025

@author: tere8
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# 1) Define absolute paths and their corresponding feature sets
datasets = {
    'playa_tmax_wspd': {
        'path': r"C:\Users\tere8\Desktop\UNIVERSIDAD\CUARTO\MODELIZACIÓN\playa_tmax_wspd.csv",
        'features': ['tmax', 'wspd']
    },
    'playa_ext_sol': {
        'path': r"C:\Users\tere8\Desktop\UNIVERSIDAD\CUARTO\MODELIZACIÓN\playa_dataset_ext_sol.csv",
        'features': ['tmax', 'wspd', 'sol']
    },
    'playa_ext_presMax': {
        'path': r"C:\Users\tere8\Desktop\UNIVERSIDAD\CUARTO\MODELIZACIÓN\playa_dataset_ext_presMax.csv",
        'features': ['tmax', 'wspd', 'sol', 'presMax']
    },
    'playa_ext_presMax_dir': {
        'path': r"C:\Users\tere8\Desktop\UNIVERSIDAD\CUARTO\MODELIZACIÓN\playa_dataset_ext_presMax_dir.csv",
        'features': ['tmax', 'wspd', 'sol', 'presMax', 'dir']
    }
}

results = []

for name, data in datasets.items():
    # Load dataset
    df = pd.read_csv(data['path'])
    
    # Ensure features and label are numeric
    for col in data['features'] + ['aptoplay']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=data['features'] + ['aptoplay'])
    
    X = df[data['features']].values
    y = df['aptoplay'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Decision Tree (entropy, max_depth=2)
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    
    # Extract tree rules
    tree_rules = export_text(clf, feature_names=data['features'])
    
    # Compute accuracies
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    
    # Print results
    print(f"Dataset: {name}")
    print("Decision Tree rules:")
    print(tree_rules)
    print(f"Train accuracy: {acc_train:.4f}")
    print(f"Test accuracy:  {acc_test:.4f}")
    print("-" * 60)
    
    results.append({
        'Dataset': name,
        'Features': ', '.join(data['features']),
        'Train Acc': round(acc_train, 4),
        'Test Acc': round(acc_test, 4),
        'Rules': tree_rules.replace('\n', ' | ')
    })

# Summary table
res_df = pd.DataFrame(results)
print("\nSummary Table:")
print(res_df[['Dataset','Features','Train Acc','Test Acc']].to_string(index=False))
