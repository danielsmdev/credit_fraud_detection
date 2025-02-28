# Detección de Fraude Financiero en Tarjetas de Crédito

## 📄 Descripción del Proyecto
Este proyecto tiene como objetivo la detección de transacciones fraudulentas en tarjetas de crédito utilizando técnicas de **Machine Learning** y **Análisis de Datos**. A través del análisis de un conjunto de datos real, se busca desarrollar un modelo que pueda identificar fraudes con alta precisión y ayudar en la toma de decisiones en entornos financieros.

## 📊 Dataset Utilizado
- **Fuente**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Descripción**: Contiene un total de **284,807 transacciones**, de las cuales solo **0.172%** son fraudulentas.
- **Características**:
  - `Time`: Tiempo en segundos desde la primera transacción registrada.
  - `Amount`: Monto de la transacción.
  - `V1` a `V28`: Variables anonimizadas obtenidas a partir de PCA.
  - `Class`: **0** para transacción legítima, **1** para transacción fraudulenta.

## 🏗️ Estructura del Proyecto
```
fraude_financiero/
│── data/               # Datos
│   ├── raw/           # Datos originales
│   ├── processed/     # Datos preprocesados
│── notebooks/         # Notebooks de Jupyter
│   ├── 01_EDA.ipynb   # Exploración de datos
│   ├── 02_Preprocessing.ipynb # Preprocesamiento de los datos
│   ├── 03_Model_Training.ipynb # Entrenamiento de los modelos de Machine Learning
│   ├── 04_Evaluation.ipynb # Evaluación de los modelos de Machine Learning
│── src/               # Código fuente
│   ├── data_prep.py   # Preprocesamiento de datos
│   ├── train_model.py # Entrenamiento del modelo
│   ├── evaluate.py    # Evaluación del modelo
│── models/            # Modelos entrenados
│── reports/           # Resultados y visualizaciones
│── requirements.txt   # Dependencias
│── README.md          # Documentación
```

## ⚙️ Instalación y Configuración
1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu_usuario/fraude_financiero.git
   cd fraude_financiero
   ```
2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```
3. **Descargar el dataset y colocarlo en `data/raw/`**

## 🔍 Análisis Exploratorio de Datos (EDA)
El análisis inicial incluye:
- Distribución de transacciones fraudulentas vs. legítimas.
- Visualización de montos de transacciones.
- Identificación de patrones en los datos.
- Matriz de correlación de variables.

## 🤖 Modelado y Evaluación
Se probarán distintos modelos para la detección de fraude:
- **Regresión Logística** (baseline)
- **Random Forest**
- **XGBoost**
- **LightGBM**
- **Redes Neuronales** (opcional)

Las métricas utilizadas para evaluar los modelos serán:
- **Precisión (`accuracy`)**
- **Recall (`recall`)** (importante en detección de fraude)
- **F1-Score**
- **Matriz de confusión**

## 📈 Resultados y Conclusiones
Se documentarán los resultados de los modelos, la importancia de las variables y las estrategias utilizadas para manejar el desbalance de clases.

## 💡 Posibles Mejoras
- Aplicación de técnicas avanzadas de balanceo de clases (SMOTE, Undersampling, etc.).
- Optimización de hiperparámetros con Grid Search o Bayesian Optimization.
- Implementación de detección de anomalías con Autoencoders.

## 📌 Contribución
Si deseas contribuir, puedes abrir un **pull request** o reportar un **issue**. 

## 📧 Contacto
Autor: **Daniel Sánchez McCambridge**
- [LinkedIn](http://www.linkedin.com/in/daniel-francisco-sanchez-mccambridge-81792b111/)
- Email: danielsm.dev@gmail.com


