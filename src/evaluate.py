import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import warnings
import logging

# Configurar la variable de entorno para silenciar el warning de joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Ajusta este valor al número de núcleos que deseas usar

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1️⃣ Cargar el modelo entrenado
def load_model(model_name):
    """Carga un modelo específico desde la carpeta models/."""
    model_path = f"../models/{model_name.lower().replace(' ', '_')}.pkl"
    try:
        logging.info(f"Cargando modelo desde {model_path}...")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo {model_name}: {e}")
        raise

# 2️⃣ Cargar los datos de test
def load_test_data():
    """Carga los datos de test desde la carpeta processed."""
    try:
        X_test = pd.read_csv("../data/processed/X_test.csv")
        y_test = pd.read_csv("../data/processed/y_test.csv")["Class"]
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error al cargar los datos de test: {e}")
        raise

# 3️⃣ Evaluar el modelo con diferentes umbrales
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evalúa el modelo con un umbral personalizado para fraude."""
    
    # Comprobar si el modelo tiene predict_proba() (algunos modelos pueden no tenerlo)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Si el modelo no tiene predict_proba(), usar predict() y convertir a 0-1 con threshold
        y_prob = model.predict(X_test)
    
    y_pred = (y_prob >= threshold).astype(int)  # Aplicar el umbral

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    logging.info("\n📌 Matriz de Confusión:")
    logging.info(cm)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"])
    logging.info("\n📌 Reporte de Clasificación:")
    logging.info(report)

    return y_pred, y_prob

# 4️⃣ Graficar la Precision-Recall Curve
def plot_precision_recall_curve(y_test, y_prob):
    """Muestra la curva Precision-Recall para encontrar el mejor umbral."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, precision[:-1], label="Precision", linestyle="--", color="blue")
    plt.plot(thresholds, recall[:-1], label="Recall", linestyle="-", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
