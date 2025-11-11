# PredictIQ – Walmart Sales Forecasting System

A full‑stack, production‑oriented sales forecasting platform that automates data ingestion, feature engineering, model training, prediction serving, and dashboard visualizations. Built to demonstrate end‑to‑end machine learning engineering and forecasting capabilities.

---

## 🚀 Project Overview

PredictIQ is an end‑to‑end **Walmart sales forecasting system** powered by **XGBoost**, served through a **FastAPI backend**, with **PostgreSQL** for data and model storage, and a responsive **HTML/CSS/JavaScript** dashboard for interacting with forecasts.

The system handles:

* Automated feature engineering (lags, rolling windows, percentage changes)
* Model training, hyperparameter tuning, and evaluation
* Continuous retraining through dynamic dataset ingestion
* REST APIs for real‑time forecasting
* Interactive web dashboard for visualization and analysis

This project highlights strong ML engineering, backend development, and system design skills.

---

## ✅ Key Features

### 🔹 **1. Automated Feature Engineering**

PredictIQ generates all high‑value forecasting features:

* Lag features (t‑1, t‑2, t‑3, etc.)
* Rolling means/standard deviations
* Percentage changes
* Date‑time decomposition (hour, day, month, year, dayofweek, etc.)

This pipeline increases signal strength and reduces manual feature work.

### 🔹 **2. XGBoost Model Training & Hyperparameter Tuning**

* Trained using historical Walmart sales data
* Automated tuning via **RandomizedSearchCV**
* RMSE evaluation for model comparison
* Pickled model storage and versioning

### 🔹 **3. Continuous Retraining Pipeline**

PredictIQ ingests new datasets and triggers:

1. Feature generation
2. Model training
3. Evaluation
4. Saving the updated model

No manual steps are required.

### 🔹 **4. FastAPI Backend**

The system exposes:

* `/predict` endpoint for single/batch predictions
* `/train` endpoint for retraining
* `/health` endpoint for service monitoring

### 🔹 **5. PostgreSQL Integration**

Stores:

* Historical and newly ingested datasets
* Model metadata
* Forecast results

### 🔹 **6. Interactive Web Dashboard (HTML/CSS/JS)**

* Displays forecasts using Chart.js
* Dynamic UI updates via API calls
* Clean and intuitive interface for analysis

---

## 🛠 Tech Stack

| Layer                 | Technology                      |
| --------------------- | ------------------------------- |
| **Forecasting Model** | XGBoost, Python, NumPy, Pandas  |
| **Feature Pipeline**  | Pandas, Custom Scripts          |
| **API Layer**         | FastAPI                         |
| **Database**          | PostgreSQL                      |
| **Frontend**          | HTML, CSS, JavaScript, Chart.js |
| **Version Control**   | Git, GitHub                     |

---

## 📁 Project Structure

```
PredictIQ/
│── data/                 # Raw and processed data
│── models/               # Saved model files (pickle)
│── backend/              # FastAPI source code
│── frontend/             # HTML/CSS/JS dashboard
│── notebooks/            # EDA and experiments
│── utils/                # Feature engineering + helpers
│── requirements.txt
│── README.md
```

---

## 🧠 Machine Learning Workflow

1. Load Walmart dataset
2. Perform automated feature engineering
3. Split into train/test
4. Train XGBoost with RandomizedSearchCV
5. Evaluate using RMSE
6. Save and version model
7. Serve with FastAPI

---

## 🔗 API Endpoints

### **POST /predict**

Returns forecast results based on input JSON.

### **POST /train**

Retrains the model using uploaded or database‑stored data.

### **GET /health**

Basic API status check.

---

## 📊 Dashboard

* View future predictions
* Analyze historical vs forecast curves
* Inspect effects of different stores/items
* Smooth UI with responsive design

---

## 📌 Resume‑Ready Description

PredictIQ – Walmart Sales Forecasting System (Python, FastAPI, PostgreSQL, HTML/CSS/JS)

* Developed an end‑to‑end sales forecasting system using XGBoost on Walmart data with automated feature engineering (lags, rolling stats, percentage changes) and hyperparameter tuning via RandomizedSearchCV.
* Implemented an automated data pipeline for continuous retraining, allowing new datasets to be ingested and models updated without manual steps.
* Integrated trained models into a FastAPI backend with PostgreSQL storage and dynamic APIs serving predictions to an interactive web dashboard.

---

## 📦 Installation

```
pip install -r requirements.txt
```

---

## ▶️ Run the App

**Start FastAPI:**

```
uvicorn backend.main:app --reload
```

**Open the dashboard:**

```
frontend/index.html
```

---

## ✅ Status

This project is complete and ready for resume/portfolio use.

---

## 📬 Contact

For questions or improvements, feel free to reach out or raise an issue.
