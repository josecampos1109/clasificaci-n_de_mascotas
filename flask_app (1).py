from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Cargar el dataset
df = pd.read_csv('/home/melchor/mysite/perros_y_gatos.csv')

# Definir características y etiquetas
X = df.drop('Especies', axis=1)
y = df['Especies']

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Crear el pipeline
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(exclude=['object']).columns

# Crear el transformador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Crear el modelo con el pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# Entrenar el modelo
model.fit(X_train, y_train)

# Función para clasificar animales
def clasificador_animal(caracteristicas):
    caracteristicas_df = pd.DataFrame(caracteristicas)
    prediccion = model.predict(caracteristicas_df)
    return prediccion[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        # Obtener datos del formulario
        caracteristicas = {
            'Comportamiento': [request.form['comportamiento']],
            'Entrenamiento': [request.form['entrenamiento']],
            'Ejercicio': [request.form['ejercicio']],
            'Comunicacion': [request.form['comunicacion']],
            'Habilidades': [request.form['habilidades']],
            'Cuidado': [request.form['cuidado']],
            'Tamaño': [request.form['tamano']],
            'Cabeza': [request.form['cabeza']],
            'Vision': [request.form['vision']],
            'Pelaje': [request.form['pelaje']],
            'Orejas': [request.form['orejas']],
            'Cola': [request.form['cola']],
            'Mandibula': [request.form['mandibula']],
            'Movimiento': [request.form['movimiento']]
        }

        # Clasificar el animal
        resultado = clasificador_animal(caracteristicas)

    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)