import pandas as pd
import itertools
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv('playa_dataset_presMax.csv')

# Crear la etiqueta aptoplay según la condición tmax ≥ 25, wspd ≤ 8, presMax > 1010
df['aptoplay'] = ((df['tmax'] >= 25) & (df['wspd'] <= 8) & (df['presMax'] > 1010)).astype(int)
df['aptoplay'] = df['aptoplay'].replace({0: -1})

# Eliminar filas con valores faltantes
df_clean = df.dropna(subset=['tmax', 'wspd', 'presMax', 'aptoplay'])

# Variables candidatas
features = ['tmax', 'wspd', 'presMax']
target = 'aptoplay'

# Probar todas las combinaciones posibles
results = []

for r in range(1, len(features)+1):
    for combo in itertools.combinations(features, r):
        X = df_clean[list(combo)]
        y = df_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearSVC(C=1.0, loss='hinge', dual=True, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results.append({
            'Variables': ', '.join(combo),
            'Train Accuracy': round(train_acc, 4),
            'Test Accuracy': round(test_acc, 4)
        })

# Mostrar resultados
results_df = pd.DataFrame(results)
print(results_df)
