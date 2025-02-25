import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 1️⃣ Cargar los datos de entrenamiento
def load_train_data():
    """Carga los datos de entrenamiento desde la carpeta processed."""
    X_train = pd.read_csv("../data/processed/X_train.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv")["Class"]  # Asegurar que se carga como una serie
    return X_train, y_train

# 2️⃣ Definir los modelos de Machine Learning
def get_models():
    """Define y devuelve varios modelos de Machine Learning."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    return models

# 3️⃣ Entrenar y evaluar modelos
def train_and_evaluate(models, X_train, y_train):
    """Entrena los modelos y evalúa su rendimiento en validación."""
    
    # División interna de train en train y validation (80-20)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🚀 Entrenando {name}...")
        model.fit(X_train_sub, y_train_sub)
        y_pred = model.predict(X_val)
        
        # Evaluar modelo en validación
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
        
        print(f"📌 {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    return results, models

# 4️⃣ Guardar el mejor modelo
def save_best_model(models, results):
    """Guarda el mejor modelo basado en la métrica F1-score."""
    best_model_name = max(results, key=lambda x: results[x]["F1-score"])  # Elegimos el modelo con mejor F1-score
    best_model = models[best_model_name]
    
    model_path = f"../models/fraud_detection_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n✅ Mejor modelo guardado: {best_model_name} en {model_path}")

# 🚀 Función principal
def main():
    """Carga datos, entrena modelos y guarda el mejor modelo."""
    X_train, y_train = load_train_data()
    models = get_models()
    results, trained_models = train_and_evaluate(models, X_train, y_train)
    save_best_model(trained_models, results)

# Si ejecutamos el script directamente, entrenamos los modelos
if __name__ == "__main__":
    main()