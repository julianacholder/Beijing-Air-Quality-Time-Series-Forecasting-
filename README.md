# Beijing Air Quality Forecasting Using Advanced LSTM

**Course**: Machine Learning Techniques I  
**Assignment**: Time Series Forecasting  
**Student**: Juliana Crystal Holder  
**Date**: May 27, 2025  
**GitHub Repository**: [Beijing-Air-Quality-Time-Series-Forecasting](https://github.com/julianacholder/Beijing-Air-Quality-Time-Series-Forecasting-.git)

---

## 📌 Table of Contents
1. [Introduction](#introduction)
2. [Data Exploration](#data-exploration)
3. [Model Design](#model-design)
4. [Experiment Results](#experiment-results)
5. [Reproducibility Instructions](#reproducibility-instructions)
6. [Repository Structure](#repository-structure)

---

## 1. Introduction

### 🏭 Problem Statement  
Air pollution, particularly PM2.5 concentrations, poses a critical threat to public health and urban planning. This project forecasts PM2.5 levels in Beijing using historical air quality and meteorological data to enable proactive mitigation measures.

### 🧠 Approach Overview  
This project employs an advanced ensemble of two LSTM-based architectures:
- **48-Hour Sequence LSTM**: Captures extended temporal patterns.
- **Wider Architecture LSTM**: Learns complex feature interactions.
- **Ensemble Integration**: Combines both models via averaging for improved prediction accuracy.

**Goal**: Achieve RMSE below 4,000 on the Kaggle test set while demonstrating systematic model experimentation.

---

## 2. Data Exploration

### 📊 Dataset Overview  
- **Training Samples**: 41,757 hourly observations  
- **Test Samples**: 13,148 hourly observations  
- **Features**: PM2.5, TEMP, PRES, DEWP, RAIN, WSPM, cbwd  
- **Period**: Multi-year hourly air quality data from Beijing  

### 📈 PM2.5 Distribution Insights  
- High variability with values from 0–994 µg/m³  
- Right-skewed distribution (many low values, few very high)  
- Seasonal patterns: Winter pollution spikes  
- Diurnal cycles: Peaks during rush hours  

### 🔄 Feature Correlations  
| Feature      | Correlation with PM2.5 |
|--------------|------------------------|
| TEMP         | -0.23 (inverse)        |
| PRES         |  0.18 (weak positive)  |
| WSPM         | -0.12 (inverse)        |
| DEWP         | Moderate correlation   |

### 🧹 Missing Data Handling  
- **Forward/Backward Fill**: Time-series aware imputation  
- **<2% Imputed**: Minimal bias introduced  

### ⚙️ Preprocessing Summary  
- **Temporal Features**: sin/cos encoding for hour, day, month  
- **Normalization**: MinMaxScaler for model stability  
- **Sequencing**: 48-hour input windows  

---

## 3. Model Design

### 🔍 Architecture Philosophy  
Combining sequence and wide LSTM architectures to capture complementary aspects of temporal and feature complexity.

---

### 🧬 Model 1: 48-Hour Sequence LSTM

**Architecture**:
- LSTM(256, `relu`, `return_sequences=True`) + Dropout(0.2)  
- LSTM(128, `relu`, `return_sequences=True`) + Dropout(0.2)  
- LSTM(64, `relu`) + Dropout(0.2)  
- Dense(1)

**Justification**:
- Captures long-range dependencies  
- Hierarchical feature extraction  
- ReLU activation for gradient stability  
- Total Parameters: **529,217**

---

### 🧬 Model 2: Wider LSTM Architecture

**Architecture**:
- LSTM(256, `relu`, `return_sequences=True`) + Dropout(0.2)  
- LSTM(128, `relu`, `return_sequences=True`) + Dropout(0.2)  
- LSTM(64, `relu`) + Dropout(0.2)  
- Dense(1)

**Justification**:
- Focuses on immediate conditions  
- Larger hidden size captures complex relationships  
- Total Parameters: **523,073**

---

### 🧪 Ensemble Methodology

**Strategy**:  
`ensemble_prediction = (model1 + model2) / 2`

**Rationale**:
- Reduces variance and bias  
- Combines temporal and feature strengths  
- Ensemble RMSE: **3,645.02 (Kaggle)**

---

### ⚙️ Optimization Configuration

| Parameter       | Value                       |
|-----------------|-----------------------------|
| Optimizer       | Adam                        |
| Learning Rate   | 0.0005                      |
| Loss Function   | Mean Squared Error          |
| Batch Sizes     | 64 (Model 1), 32 (Model 2)  |
| Early Stopping  | Patience = 10               |
| Validation Split| 15% (M1), 20% (M2)          |

**Note**: Hyperparameters were tuned across **15+ experiments** targeting a validation RMSE "convergence zone" (0.083–0.088) for generalization.

---

## 4. Experiment Results

| Exp | Architecture              | Optimizer | LR     | Batch | Epochs | Dropout | Val RMSE | Kaggle RMSE |
|-----|---------------------------|-----------|--------|-------|--------|---------|----------|-------------|
| 1   | 2-layer LSTM (64→32)      | Adam      | 0.001  | 32    | 50     | 0.2     | 0.0800   | 4,767.11    |
| 2   | 3-layer LSTM (128-64-32)  | Adam      | 0.0005 | 32    | 75     | 0.2     | 0.0830   | 4,646.72    |
| 3   | 3-layer LSTM              | Adam      | 0.0001 | 32    | 75     | 0.2     | 0.0758   | 4,939.57    |
| 4   | 3-layer LSTM              | RMSprop   | 0.0005 | 32    | 75     | 0.2     | 0.0755   | 5,843.38    |
| 5   | 3-layer LSTM              | Adam      | 0.0005 | 16    | 75     | 0.2     | 0.0832   | 4,549.76    |
| ... | ...                       | ...       | ...    | ...   | ...    | ...     | ...      | ...         |
| 13  | 3-layer LSTM (seq=48h)    | Adam      | 0.0005 | 64    | 11     | 0.2     | 0.0875   | 4,496.82    |
| 15  | **Ensemble (Model 2g+8)**| Combined  | -      | -     | -      | -       | -        | **3,645.02** |

### 🧠 Key Insights
- **Convergence Zone (0.083–0.088)** led to best generalization
- **Low validation RMSE (<0.083)** correlated with **Kaggle overfitting**
- **48-hour sequences** outperformed shallow/single-step inputs
- **Ensemble approach** achieved best public leaderboard score

---

## 5. Reproducibility Instructions

### ⚙️ Requirements
```bash
pip install -r requirements.txt

```

##  6. Run

```bash
# Preprocess data
python scripts/preprocess.py

# Train Model 1
python scripts/train_model_48hr.py

# Train Model 2
python scripts/train_model_wide.py

# Generate Ensemble Predictions
python scripts/ensemble_predict.py

# Evaluate final results
python scripts/evaluate.py
```

📁 Beijing-Air-Quality-Time-Series-Forecasting
│
├── 📁 data/              # Raw and processed datasets
├── 📁 notebook/          
├── 📁 outputs/           # Predictions, visualizations, and logs
└── README.md             # Project overview and instructions




