# Fraud Detection System for E-Commerce and Banking Transactions

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-LightGBM%2C%20XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents
- [Fraud Detection System for E-Commerce and Banking Transactions](#fraud-detection-system-for-e-commerce-and-banking-transactions)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Key Features](#key-features)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [1. Data Processing](#1-data-processing)
    - [2. Model Training](#2-model-training)
    - [3. Real-time Prediction API](#3-real-time-prediction-api)
  - [Data Description](#data-description)
    - [E-Commerce Dataset (`Fraud_Data.csv`)](#e-commerce-dataset-fraud_datacsv)
    - [Banking Dataset (`creditcard.csv`)](#banking-dataset-creditcardcsv)
  - [Model Approach](#model-approach)
  - [Performance Metrics](#performance-metrics)
  - [API Documentation](#api-documentation)
    - [Request:](#request)
    - [Response:](#response)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

---

## Project Overview

This project develops a machine learning system to detect fraudulent transactions in both e-commerce and banking domains. The solution addresses critical business needs by:

- Reducing false positives that impact customer experience
- Minimizing false negatives that lead to financial losses
- Providing real-time fraud probability scoring
- Offering model interpretability for investigators

**Business Impact**: Can reduce fraud-related losses by 30–50% while maintaining <5% false positive rates based on industry benchmarks.

---

## Key Features

- ✅ **Dual-domain detection**: Handles both e-commerce and banking patterns  
- 🌍 **Geolocation intelligence**: IP-based anomaly detection  
- ⏱ **Behavioral profiling**: Velocity & time-based features  
- ⚙️ **Production-ready API**: Flask-based real-time prediction  
- 🔍 **Explainable AI**: SHAP-based fraud reason codes  
- 🔁 **CI/CD Pipeline**: Automated testing & deployment

---

## Repository Structure

```
fraud-detection-ml/
├── config/
│   ├── business_rules.json
│   ├── pipeline_config.yaml
│   └── constants.py
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   ├── creditcard.csv
│   │   └── IpAddress_to_Country.csv
│   └── processed/
│       ├── merged_transactions.parquet
│       └── normalized_data.parquet
├── docs/
│   ├── business_logic.md
│   ├── model_interpretation/
│   │   └── shap_summary_plots.html
│   └── EDA_findings.md
├── models/
│   ├── logistic_regression.pkl
│   ├── xgboost_model.pkl
│   └── performance_metrics.json
├── notebooks/
│   ├── 01_eda_fraud_detection.ipynb
│   └── 02_feature_engineering.ipynb
├── scripts/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── explain_model.py
├── tests/
│   ├── unit/
│   │   ├── test_data_processing.py
│   │   └── test_feature_engineering.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_business_rules.py
├── api/
│   └── app.py
├── .env.example
├── requirements.txt
├── Makefile
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip 20.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/fraud-detection-ml.git
cd fraud-detection-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Data Processing

```python
from scripts.data_processing import clean_fraud_data
from scripts.feature_engineering import FraudFeatureEngineer

# Load and clean data
df = clean_fraud_data('data/raw/Fraud_Data.csv')

# Feature engineering
engineer = FraudFeatureEngineer()
features = engineer.transform(df)
```

### 2. Model Training

```bash
python scripts/train_model.py   --data_path data/processed/features.csv   --model_output models/xgboost_v1.pkl
```

### 3. Real-time Prediction API

```bash
flask run --app api/app.py
```

Example API Request:

```json
{
  "transaction_amount": 249.99,
  "time_since_signup": 86400,
  "country": "US",
  "device_id": "AXB1245"
}
```

---

## Data Description

### E-Commerce Dataset (`Fraud_Data.csv`)

| Feature             | Description                              |
|---------------------|------------------------------------------|
| purchase_value      | Transaction amount in USD                |
| time_since_signup   | Seconds between signup and purchase      |
| device_id           | Unique device identifier                 |
| country             | Derived from IP address                  |
| class (target)      | 1 = Fraud, 0 = Legitimate                |

### Banking Dataset (`creditcard.csv`)

| Feature   | Description                                |
|-----------|--------------------------------------------|
| V1–V28    | PCA-transformed anonymized features        |
| Amount    | Transaction amount                         |
| Class     | 1 = Fraud, 0 = Legitimate                  |

---

## Model Approach

**Ensemble Architecture** using:

- SMOTE for class imbalance
- Recursive Feature Elimination (RFE)
- Bayesian Optimization for hyperparameter tuning
- Threshold optimization using precision-recall tradeoffs

---

## Performance Metrics

| Model         | Precision | Recall | F1-Score | AUC-PR |
|---------------|-----------|--------|----------|--------|
| XGBoost (v1)  | 0.92      | 0.85   | 0.88     | 0.93   |
| Logistic Reg. | 0.76      | 0.68   | 0.72     | 0.75   |

**Confusion Matrix (Threshold = 0.6):**

```
              Predicted
           |   0   |   1
Actual  0  | 9823  | 177
        1  |  83   | 917
```

---

## API Documentation

**POST** `/predict`

### Request:

```json
{
  "transaction_amount": float,
  "time_since_signup": int,
  "device_id": "string",
  "browser": "string",
  "country": "string"
}
```

### Response:

```json
{
  "fraud_probability": 0.87,
  "shap_values": {
    "top_features": [
      {"feature": "purchase_value", "value": 0.32},
      {"feature": "time_since_signup", "value": 0.21}
    ]
  }
}
```

---

## Contributing

1. Fork the repository  
2. Create your branch: `git checkout -b feature/your-feature`  
3. Commit your changes: `git commit -am 'Add some feature'`  
4. Push to the branch: `git push origin feature/your-feature`  
5. Open a Pull Request  

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Project Lead**:  
**Email**: yourname@domain.com  
**Company**: Your Company Name  
