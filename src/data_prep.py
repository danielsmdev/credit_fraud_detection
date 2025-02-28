import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 1️⃣ Cargar los datos
def load_data(filepath):
    """Carga el dataset desde un archivo CSV."""
    df = pd.read_csv(filepath)
    return df

# 2️⃣ Selección de Variables (Feature Selection)
def select_features(df):
    """Selecciona las variables más relevantes según el análisis de Feature Importance."""
    important_features = [
        "V17", "V12", "V14", "V10", "V11", "V16", "V9", "V18", "V4", "V7", "Amount"
    ]
    df = df[important_features + ["Class"]]  # Mantener solo estas variables + la variable objetivo
    return df

# 3️⃣ Transformaciones (Escalado y Logaritmo)
def transform_features(df):
    """Aplica transformaciones a las variables necesarias."""
    df["Amount_log"] = np.log1p(df["Amount"])  # Aplicamos log(Amount + 1)
    df.drop(columns=["Amount"], inplace=True)  # Eliminamos la columna original
    return df

# 4️⃣ División en Train/Test (ANTES de aplicar SMOTE)
def split_data(df):
    """Divide los datos en entrenamiento (80%) y prueba (20%) antes de aplicar SMOTE."""
    X = df.drop(columns=["Class"])  # Features
    y = df["Class"]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

# 5️⃣ Aplicar SMOTE solo a Train
def balance_data(X_train, y_train):
    """Aplica SMOTE para balancear las clases en el conjunto de entrenamiento."""
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

# 🚀 Función principal para procesar los datos
def prepare_data(filepath):
    """Función completa para preparar los datos para el modelo."""
    df = load_data(filepath)
    df = select_features(df)
    df = transform_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test

# Guardar datasets transformados en la carpeta data/processed/
def save_processed_data(X_train, X_test, y_train, y_test):
    """Guarda los datos procesados en archivos CSV."""
    X_train.to_csv("../data/processed/X_train.csv", index=False)
    X_test.to_csv("../data/processed/X_test.csv", index=False)

    # Guardar etiquetas asegurando que la columna se llama "Class"
    y_train.to_frame(name="Class").to_csv("../data/processed/y_train.csv", index=False)
    y_test.to_frame(name="Class").to_csv("../data/processed/y_test.csv", index=False)

    print("✔️ Datos procesados guardados en data/processed/")

# Si ejecutamos el script directamente, procesamos y guardamos los datos
if __name__ == "__main__":
    data_path = "../data/raw/creditcard.csv"
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    save_processed_data(X_train, X_test, y_train, y_test)
    print(f"✔️ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
