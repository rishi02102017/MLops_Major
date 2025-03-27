
# MLOps Pipeline for Iris Dataset

A complete MLOps workflow for **Iris Flower Classification** using **scikit-learn**, **MLflow**, and **Docker**.

---

## 📌 Overview

This project demonstrates a structured MLOps pipeline using the Iris dataset, covering:

- ✅ Data processing using custom classes  
- ✅ Experiment tracking with MLflow  
- ✅ Model quantization and testing  
- ✅ Docker-based containerization and deployment  

---

## 📂 Dataset

- **Source:** `sklearn.datasets.load_iris()`
- **Classes:** Setosa, Versicolor, Virginica
- **Features:** Sepal length, Sepal width, Petal length, Petal width

---

## 🔧 Project Structure

```bash
.
├── ML_Ops_Major.py          # Main Python script with full pipeline
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image setup
├── docker-compose.yml       # Compose config for MLflow + App
└── README.md                # You're reading it!
```

---

## 🧠 Q1: Data Processing Class

Implemented a class `IrisDataProcessor` that:

- Converts raw data to a pandas DataFrame  
- Applies feature scaling with `StandardScaler`  
- Performs a `train_test_split()`  
- Returns basic statistical summary via `.describe()`  

```python
processor = IrisDataProcessor()
X_train, X_test, y_train, y_test = processor.prepare_data()
print(processor.get_feature_stats())
```

---

## 🧪 Q2: Experiment Tracking with MLflow

Implemented `IrisExperiment` class to:

- Train **Logistic Regression** and **Random Forest**  
- Track experiments using **MLflow**
- Log **accuracy**, **precision**, and **recall**
- Save trained models with `mlflow.sklearn.log_model()`

```python
experiment = IrisExperiment(processor)
experiment.run_experiment()
```

🎯 Results are viewable in the MLflow UI (`mlflow ui`).

---

## 🛠 Q3: Model Optimization & Testing

Created `IrisModelOptimizer` class to:

- Quantize (serialize) the Logistic Regression model using `joblib`
- Run **basic unit tests** on the saved model

```python
optimizer = IrisModelOptimizer(experiment)
optimizer.quantize_model()
optimizer.run_tests()
```

---

## 📦 Q4: Dockerization

### 🐳 Dockerfile

- Base: `python:3.9-slim`
- Installs all dependencies via `requirements.txt`
- Secure and lightweight image

### 🛠 docker-compose.yml

- Spins up both MLflow and the ML app
- Manages volumes, ports, and environment
- Allows one-command deployment:

```bash
docker-compose build
docker-compose up
```

---

## 🚀 Expected Output

✅ Processed and scaled dataset  
✅ MLflow-tracked experiment metrics & models  
✅ Serialized (quantized) logistic regression model  
✅ All unit tests passed  
✅ Dockerized end-to-end workflow  

---

## 📊 Sample MLflow Dashboard

![MLflow Comparison](./screenshots/compare_runs.png)
![MLflow Metrics](./screenshots/metrics_table.png)
![Docker Run](./screenshots/docker_up.png)

---

## 🧰 Tech Stack

| Tool          | Purpose                        |
|---------------|--------------------------------|
| Python        | Core language                  |
| pandas, numpy | Data processing                |
| scikit-learn  | ML models & preprocessing      |
| MLflow        | Experiment tracking            |
| joblib        | Model serialization            |
| Docker        | Containerization               |

---

## 📎 Requirements

```text
pandas
numpy
joblib
scikit-learn
mlflow
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧪 How to Run Locally

1. Clone the repo
2. Build & start services:
   ```bash
   docker-compose build
   docker-compose up
   ```
3. Access MLflow UI at `http://127.0.0.1:5000`
4. Run `ML_Ops_Major.py` using Python or Colab

---

## ✨ Author
- **Jyotishman Das**  
- M.Tech AI @ IIT Jodhpur | AI/ML Engineer  
- [LinkedIn](https://www.linkedin.com/in/jyotishmandas85p)
- [Website](https://my-portfolio-jyotishman-das-projects.vercel.app/)
