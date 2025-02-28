import pandas as pd
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import os
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # O ajusta al número de núcleos reales de tu CPU

# 1️⃣ Cargar los datos de entrenamiento
def load_train_data():
    """Carga los datos de entrenamiento desde la carpeta processed."""
    try:
        X_train = pd.read_csv("../data/processed/X_train.csv")
        y_train = pd.read_csv("../data/processed/y_train.csv")["Class"]  # Asegurar que se carga como una serie
        return X_train, y_train
    except Exception as e:
        logging.error(f"Error al cargar los datos de entrenamiento: {e}")
        raise

def optimize_xgboost(X_train, y_train):
    """Optimiza los hiperparámetros de XGBoost con GridSearchCV."""
    
    param_grid = {
        "n_estimators": [100, 200],  # Reducido para evitar largas esperas
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",  # 🔹 Evita el warning
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,  # 🔹 Validación cruzada con 3 folds
        verbose=1,  # 🔹 Reduce la cantidad de logs
        n_jobs=-1  # 🔹 Usa todos los núcleos disponibles
    )
    
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Mejores Hiperparámetros para XGBoost: {grid_search.best_params_}")
    logging.info(f"Mejor F1-score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# 2️⃣ Definir los modelos de Machine Learning
def get_models():
    """Define y devuelve varios modelos de Machine Learning."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    return models

# 3️⃣ Entrenar y evaluar modelos
def train_and_evaluate(models, X_train, y_train):
    """Entrena los modelos y evalúa su rendimiento."""
    
    results = {}

    for name, model in models.items():
        logging.info(f"Entrenando {name}...")
        
        if name == "XGBoost":
            logging.info("Optimizando hiperparámetros de XGBoost...")
            model = optimize_xgboost(X_train, y_train)  # Aplicar GridSearchCV
        
        model.fit(X_train, y_train)
        
        f1_score_train = f1_score(y_train, model.predict(X_train))
        results[name] = {"F1-score": f1_score_train}
        
        logging.info(f"{name} - F1-score: {f1_score_train:.4f}")

    return results, models

# 4️⃣ Guardar el mejor modelo
def save_all_models(models, results, X_train, y_train):
    """Guarda todos los modelos entrenados en la carpeta models/."""
    best_model_name = max(results, key=lambda x: results[x]["F1-score"])  # Encuentra el mejor modelo
    logging.info(f"Mejor modelo basado en F1-score: {best_model_name}")

    for name, model in models.items():
        model_path = f"../models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        logging.info(f"Modelo guardado: {name} en {model_path}")

    # 🔹 Asegurar que XGBoost optimizado se guarda correctamente con `fit()`
    if "XGBoost" in models:
        xgb_model = models["XGBoost"]
        xgb_model.fit(X_train, y_train)  # Asegurar que el modelo se ajusta antes de guardarlo
        joblib.dump(xgb_model, "../models/xgboost.pkl")
        logging.info("Modelo XGBoost optimizado guardado correctamente.")

# 🚀 Función principal
def main():
    """Carga datos, entrena modelos y guarda el mejor modelo."""
    X_train, y_train = load_train_data()
    models = get_models()
    results, trained_models = train_and_evaluate(models, X_train, y_train)
    save_all_models(trained_models, results, X_train, y_train)

# Si ejecutamos el script directamente, entrenamos los modelos
if __name__ == "__main__":
    main()