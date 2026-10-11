# Importar librerías
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn


# Función para cargar el dataset
def load_data():
    data_path = Path(__file__).resolve().parent / "data" / "winequality_white.csv"

    df = pd.read_csv(data_path, sep=";")

    return df

# Función para preparar los datos
def preprocess_data(df, seed=42):

    # Separar variables predictoras y variable objetivo
    X = df.drop(columns=["quality"])
    y = df["quality"]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )

    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo
def train_model(X_train, y_train, alpha=1.0, max_iter=1000, tol=0.0001):

    # Crear pipeline de escalado y regresión Ridge
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            solver="lsqr"
        ))
    ])

    # Entrenar el modelo
    model.fit(X_train, y_train)

    return model

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test):

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de evaluación
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2

# Función para recibir argumentos
def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Entrenamiento de Ridge Regression con MLflow"
    )

    # Hiperparámetros del modelo
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=0.0001)

    # Semilla aleatoria
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

# Función para ejecutar y registrar un experimento
def run_experiment(args):

    # Configurar el experimento de MLflow
    mlflow.set_experiment("Wine_Quality_Ridge_Experiments")

    # Iniciar una corrida
    with mlflow.start_run():

        # Cargar el dataset
        df = load_data()

        # Preparar los datos
        X_train, X_test, y_train, y_test = preprocess_data(
            df,
            seed=args.seed
        )

        # Entrenar el modelo
        model = train_model(
            X_train,
            y_train,
            alpha=args.alpha,
            max_iter=args.max_iter,
            tol=args.tol
        )

        # Evaluar el modelo
        mae, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Registrar hiperparámetros
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("tol", args.tol)
        mlflow.log_param("seed", args.seed)

        # Registrar métricas
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Crear archivo con los resultados
        with open("metrics.txt", "w") as file:
            file.write(f"Alpha: {args.alpha}\n")
            file.write(f"Max iterations: {args.max_iter}\n")
            file.write(f"Tolerance: {args.tol}\n")
            file.write(f"Seed: {args.seed}\n")
            file.write(f"MAE: {mae:.4f}\n")
            file.write(f"RMSE: {rmse:.4f}\n")
            file.write(f"R2: {r2:.4f}\n")

        # Registrar el archivo como artefacto
        mlflow.log_artifact("metrics.txt")

        # Registrar el modelo entrenado
        mlflow.sklearn.log_model(
            sk_model=model,
            name="ridge_model"
        )

        # Mostrar resultados en la Terminal
        print("Modelo Ridge entrenado correctamente")
        print(f"Alpha: {args.alpha}")
        print(f"Max iterations: {args.max_iter}")
        print(f"Tolerance: {args.tol}")
        print(f"Seed: {args.seed}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

# Ejecutar el experimento
if __name__ == "__main__":

    # Obtener argumentos de la Terminal
    args = parse_arguments()

    # Ejecutar entrenamiento y registro en MLflow
    run_experiment(args)
