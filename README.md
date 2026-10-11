# Experimentación con MLflow - Wine Quality
### Carolina Nicole González Leal A01284948

## Descripción

Este proyecto implementa un flujo de experimentación para predecir la calidad del vino blanco utilizando un modelo de regresión Ridge.

El código permite cargar y preparar los datos, entrenar el modelo, evaluar su desempeño y registrar los resultados de cada ejecución mediante MLflow.

Se realizaron experimentos modificando tres hiperparámetros del modelo: `alpha`, `max_iter` y `tol`, con el objetivo de comparar sus efectos sobre las métricas de evaluación.

## Dataset

Se utiliza el dataset Wine Quality (White Wine), que contiene 4,898 registros y 12 variables.

La variable objetivo es `quality`, mientras que las otras 11 variables se utilizan como características para realizar las predicciones.

El archivo se encuentra en:

`data/winequality_white.csv`

## Requisitos

- Python 3.13.15
- Las librerías indicadas en `requirements.txt`

## Instalación

1. Clonar el repositorio y acceder a la carpeta del proyecto.

2. Crear un entorno virtual:

```bash
python3.13 -m venv .venv
```

3. Activar el entorno virtual (macOS/Linux):

```bash
source .venv/bin/activate
```

4. Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Para ejecutar el modelo con los valores predeterminados:

```bash
python train.py
```

También es posible modificar los hiperparámetros desde la terminal:

```bash
python train.py --alpha 0.01 --max_iter 20 --tol 0.0001 --seed 42
```

### Hiperparámetros

| Parámetro | Descripción | Valor predeterminado |
|---|---|---|
| `alpha` | Intensidad de regularización del modelo Ridge | 1.0 |
| `max_iter` | Número máximo de iteraciones del algoritmo | 1000 |
| `tol` | Tolerancia utilizada como criterio de convergencia | 0.0001 |
| `seed` | Semilla aleatoria para reproducir la división de los datos | 42 |

## Registro de experimentos

Los experimentos se registran automáticamente en MLflow bajo el nombre:

`Wine_Quality_Ridge_Experiments`

Cada ejecución registra:

- Los hiperparámetros utilizados.
- Las métricas MAE, RMSE y R².
- Un archivo `metrics.txt` con los resultados.
- El modelo entrenado.

Para visualizar los experimentos, ejecutar desde la carpeta principal del proyecto:

```bash
mlflow ui
```

Después, abrir la dirección local que indique la terminal, normalmente `http://127.0.0.1:5000`.

## Resultados

Se evaluaron diez configuraciones diferentes del modelo Ridge.

La configuración seleccionada fue:

- `alpha = 0.01`
- `max_iter = 20`
- `tol = 0.0001`
- `seed = 42`

Sus resultados fueron:

| Métrica | Resultado |
|---|---:|
| MAE | 0.5863 |
| RMSE | 0.7543 |
| R² | 0.2653 |

Esta configuración obtuvo uno de los mejores desempeños observados y produjo las mismas métricas que una ejecución equivalente con `max_iter = 1000`.

Los resultados completos y su análisis se presentan en el reporte.