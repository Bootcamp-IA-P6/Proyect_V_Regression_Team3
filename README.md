# 🐾 PetAdopt Predictor

> **Predicting how long an animal will stay in a shelter before being adopted.**  
> A machine learning pipeline built on real data from the Austin Animal Center.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-13%20passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Docker](#-docker)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Testing](#-testing)
- [Team](#-team)
- [Resumen en Español](#-resumen-en-español)

---

## 🎯 Overview

Animal shelters struggle to allocate resources efficiently because they cannot predict which animals will be adopted quickly and which will need extra attention and promotion.

**PetAdopt Predictor** solves this by using historical intake data to estimate — at the moment of arrival — how many days an animal is likely to spend in the shelter before adoption. This enables shelter staff to act proactively rather than reactively.

### Key findings
- **Breed is the strongest predictor**: Pit Bull Mix dogs take ~3x longer to be adopted than Dachshund Mix dogs.
- **Age alone is a weak predictor** (Pearson r = 0.09): the relationship is complex and non-linear.
- **XGBoost outperforms all other models** tested, with a MAE of ~30 days on unseen data.
- **The model is stable**: K-Fold CV (k=5) confirmed RMSE of 1.0786 ± 0.0058 across all folds.

---

## 🖥️ Demo

The app accepts animal characteristics at intake and returns an estimated adoption time with a risk classification:

| Risk Level | Estimated Days | Action |
|---|---|---|
| 🟢 Fast adoption | ≤ 14 days | Standard care |
| 🟡 Moderate | 15–45 days | Monitor closely |
| 🔴 High risk | > 45 days | Activate promotion protocols |

---

## 📊 Results

### Model Comparison (Validation Set)

| Model | Val R² | Val RMSE | MAE (days) | Overfitting |
|---|---|---|---|---|
| LinearRegression | 0.1464 | 1.1457 | 32.5 | −0.0015 |
| DecisionTree | 0.0844 | 1.1866 | 32.6 | 0.3481 ⚠️ |
| Ridge | 0.1465 | 1.1456 | 32.5 | −0.0016 |
| RandomForest | 0.2465 | 1.0764 | 29.9 | 0.0317 |
| **XGBoost ✅** | **0.2540** | **1.0711** | **29.7** | **0.0454** |

### Final Evaluation on Test Set (XGBoost)

| Metric | Value |
|---|---|
| R² | 0.2381 |
| MAE (log scale) | 0.8574 |
| RMSE (log scale) | 1.0902 |
| **MAE (real days)** | **30.5 days** |
| RMSE (real days) | 61.2 days |

### Cross-Validation (K-Fold, k=5)

| Fold | RMSE | R² | MAE |
|---|---|---|---|
| 1 | 1.0820 | 0.2449 | 0.8513 |
| 2 | 1.0714 | 0.2507 | 0.8472 |
| 3 | 1.0719 | 0.2617 | 0.8489 |
| 4 | 1.0818 | 0.2484 | 0.8526 |
| 5 | 1.0857 | 0.2478 | 0.8549 |
| **Mean** | **1.0786 ± 0.0058** | **0.2507 ± 0.0058** | **0.8510 ± 0.0027** |

### Top Features (XGBoost)

| Feature | Importance |
|---|---|
| Breed_grouped_Pit Bull Mix | 0.1645 |
| AnimalType_Cat | 0.0862 |
| Breed_grouped_Other | 0.0493 |
| Breed_grouped_Chihuahua Shorthair Mix | 0.0459 |
| AgeInDays | 0.0438 |

> The model reflects a real societal bias: Pit Bull and Staffordshire mixes face systematic adoption barriers that are captured and quantified by the model.

---

## 📁 Project Structure

```
Proyect_V_Regression_Team3/
│
├── 📓 notebooks/
│   ├── 00_data_preparation.ipynb    # Raw data ingestion and initial cleaning
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_modelado.ipynb            # Full modeling pipeline (baseline + advanced)
│   ├── 03_cross_validation.ipynb    # K-Fold CV (k=5) on XGBoost
│   └── 04_optuna.ipynb              # Bayesian hyperparameter optimization (50 trials)
│   └── 05_model_evaluation.ipynb    # Final evaluation of the model and conclusions
│
├── 📊 data/
│   ├── pet_adoption_model.csv       # Cleaned dataset (52,535 rows, 0 nulls)
│   ├── cv_results.csv               # K-Fold results by fold
│   └── best_hyperparams.csv         # Best hyperparameters from Optuna
│
├── 🤖 models/
│   ├── best_model_XGBoost.pkl       # Production model (serialized pipeline)
│   └── optimized_model.pkl          # Optuna-optimized model (not deployed)
│
├── 🌐 src/app/
│   ├── streamlit_app.py             # Main Streamlit application
│   ├── supabase_client.py           # Supabase connection handler
│   └── database.py                  # Prediction storage logic
│
├── 🧪 tests/
│   ├── test_baseline_metrics.py     # Baseline model validation
│   ├── test_ensemble_metrics.py     # XGBoost model validation
│   ├── test_preprocessing.py        # Data pipeline validation
│   ├── test_model_loading.py        # Model serialization checks
│   └── test_prediction_consistency.py  # Output consistency checks
│
├── 🐳 docker/
│   └── Dockerfile                   # Production container definition
│
├── docker-compose.yml               # Full environment orchestration
├── .dockerignore                    # Docker build exclusions
├── .env.example                     # Environment variables template
├── .streamlit/secrets.toml.example  # Streamlit secrets template
└── pyproject.toml                   # Project dependencies (uv)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| ML Framework | XGBoost, Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Database | Supabase (PostgreSQL) |
| Containerization | Docker, Docker Compose |
| Package Manager | uv |
| Hyperparameter Optimization | Optuna |
| Testing | pytest |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose (for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Bootcamp-IA-P6/Proyect_V_Regression_Team3.git
cd Proyect_V_Regression_Team3

# Install dependencies
uv sync

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your Supabase credentials

# Run the app
uv run streamlit run src/app/streamlit_app.py
```

The app will be available at `http://localhost:8501`.

### Running the Notebooks

```bash
uv run jupyter notebook notebooks/
```

Execute notebooks in order:
1. `00_data_preparation.ipynb`
2. `01_eda.ipynb`
3. `02_modelado.ipynb`
4. `03_cross_validation.ipynb`
5. `04_optuna.ipynb`
6. `05_model_evaluation.ipynb`

---

## 🐳 Docker

### Quick Start

```bash
# 1. Configure credentials
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Fill in your Supabase URL and key

# 2. Build and run
docker-compose up --build

# 3. Access the app
# http://localhost:8501
```

### Useful Commands

```bash
# Run in background
docker-compose up -d

# View logs
docker-compose logs -f streamlit

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up --build
```

### What's included in the container

| Included | Excluded |
|---|---|
| `src/app/` (all app files) | `notebooks/` |
| `models/best_model_XGBoost.pkl` | `data/` CSV files |
| Runtime dependencies | Dev dependencies (Jupyter, etc.) |
| — | `models/optimized_model.pkl` |

> **Security note:** Never commit `.streamlit/secrets.toml` or `.env` to Git. Both are listed in `.gitignore`. Use the `.example` files as templates.

---

## 🧠 Machine Learning Pipeline

### Data Cleaning

| Step | Before | After |
|---|---|---|
| Total records | 59,919 | 52,535 |
| Duplicate records | 6,601 | 0 |
| Null values | — | 0 |
| Unique breeds | 1,884 | 26 (top 25 + Other) |
| Target skewness | 2.685 | 0.298 (after log1p) |

### Feature Engineering

- **Breed_grouped**: Top 25 breeds by frequency; all others → "Other"
- **breed_type**: Binary — purebred / mix
- **Color_grouped**: Monocolor / Bicolor / Tricolor
- **AgeGroup**: 5 ordinal buckets (Cachorro / Joven / Adulto joven / Adulto / Senior)
- **Target**: `log1p(TimeInShelterDays)` — predictions converted back with `expm1()`

### Preprocessing Pipeline

```python
ColumnTransformer([
    ("ohe",     OneHotEncoder(handle_unknown="ignore"),    categorical_cols),
    ("ordinal", OrdinalEncoder(categories=age_order),      ["AgeGroup"]),
    ("scaler",  StandardScaler(),                          ["AgeInDays"])
])
```

### Hyperparameter Optimization

Bayesian optimization via Optuna (50 trials) was applied to XGBoost. The default configuration outperformed the optimized one (RMSE: 1.0711 vs 1.0775), indicating that the model's performance ceiling is data-limited rather than hyperparameter-limited.

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific module
uv run pytest tests/test_ensemble_metrics.py -v
```

### Test Coverage

| File | Tests | What it validates |
|---|---|---|
| `test_baseline_metrics.py` | 3 | Baseline model R², RMSE, serialization |
| `test_ensemble_metrics.py` | 1 | XGBoost R², RMSE, overfitting threshold |
| `test_preprocessing.py` | 4 | Output shape, nulls, encoding, scaling |
| `test_model_loading.py` | 2 | File existence, successful deserialization |
| `test_prediction_consistency.py` | 3 | Output shape, no NaN, non-negative days |
| **Total** | **13** | **13 passed ✅** |

---

## 👥 Team

**Proyecto V — Team 3** · 2026

| Member | Role |
|---|---|
| Raúl | Cross-validation, Optuna hiperparameters, Docker |
| Maryori | Modeling pipeline, Supabase integration, Database schema |
| Michelle | Modeling baseline, Unit testing, CI validation |
| Jose-Julio | Preprocessing data, Data exploratory analysis |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🇪🇸 Resumen en Español

**PetAdopt Predictor** es un sistema de predicción del tiempo de adopción de animales en refugios, desarrollado con datos reales del Austin Animal Center.

### ¿Qué hace?
Predice cuántos días tardará un animal en ser adoptado en el momento de su ingreso al refugio, permitiendo al personal actuar de forma preventiva.

### Pipeline completo
- **EDA**: Análisis exploratorio con transformación logarítmica de la variable objetivo
- **Modelado**: Comparativa de 5 modelos; XGBoost ganador con MAE de 30.5 días en test
- **Validación**: K-Fold (k=5) confirmando estabilidad del modelo (RMSE ± 0.0058)
- **Optimización**: Optuna con 50 trials — el modelo por defecto superó al optimizado
- **App**: Interfaz Streamlit con predicción en tiempo real y almacenamiento en Supabase
- **Docker**: Contenedor de producción listo para despliegue

### Cómo ejecutar localmente

```bash
uv sync
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Rellenar credenciales de Supabase en secrets.toml
uv run streamlit run src/app/streamlit_app.py
```

### Con Docker

```bash
docker-compose up --build
# App disponible en http://localhost:8501
```
