# Deep Learning Equity Trend Predictor

Context: (Under the weak-form Efficient Market Hypothesis (EMH), asset prices fully reflect all historical price and volume information, implying that future price movements cannot be systematically predicted using past market data.)

An empirical machine learning pipeline evaluating the effectiveness of a deep learning model versus weak-form EMH aligned baselines across the MAGMA companies (Microsoft Amazon Google Meta Apple). This project benchmarks a custom arcitechtured multi-layer stacked Long Short-Term Memory (LSTM) network against a Random Forest ensemble to evaluate whether technical indicators and temporal features contain exploitable directional signal.

---

## Project Overview

This repository implements an end-to-end time-series pipeline to test this premise:
- Extracts and preprocesses historical price, return, and volume series for large-cap tech equities.
- Adds rolling statistical features and cyclical temporal encodings.
- Evaluates multi-step recursive forecasting performance across both deep learning and tree-based baseline models.

---

## Features & Engineering

- **3D Input Tensors:** Constructed rolling 60-day sequence windows formatted as $(N, 60, F)$ tensors for sequence modeling.
- **Feature Engineering:**
  - Return momentum and rolling volatility metrics.
  - Normalized volume shift indicators.
  - Cyclical sine/cosine time encoding for calendar features (day of week, month of year) to capture seasonal market patterns without introducing artificial boundary discontinuities.
- **Data Leakage Auditing:** Strict temporal train/test split enforcing expanding or rolling validation without future look-ahead contamination.

---

## Model Architectures

1. **Random Forest Baseline (`Scikit-learn`):**
   - Ensemble of decision trees trained on flattened rolling window features.
   - Evaluated as a structural benchmark against non-linear sequence modeling.

2. **Stacked LSTM Network (`TensorFlow / Keras`):**
   - 2-layer stacked Long Short-Term Memory architecture.
   - Regularized with **30% dropout** between layers to prevent overfitting to historical market noise.
   - Trained using Adam optimizer with Mean Squared Error (MSE) loss.

---

## 📊 Empirical Results & Key Findings

Models were evaluated out-of-sample using **Normalized Root Mean Squared Error (NRMSE)** and **Directional Accuracy**:


| Model Architecture | Out-of-Sample Directional Accuracy |
| :--- | :---: |
| **Random Forest (Baseline)** | **55%** |
| **2-Layer Stacked LSTM** | **53%** |

| Model Architecture | Out-of-Sample NRMSE |
| :--- | :---: |
| **Random Forest (Baseline)** | **0.06** |
| **2-Layer Stacked LSTM** | **0.035** |


### Key Observations
- **Baseline Superiority:** The Random Forest baseline slightly outperformed the deep learning architecture on directional classification (55% vs. 53%).
- **Mean-Reversion Tendencies:** The LSTM network showed a tendency to collapse toward mean-reverting predictions due to the low signal-to-noise ratio in financial price returns.
- **Recursive Multi-Step Compounding:** During 10-day recursive forecasts, autoregressive error propagation rapidly degraded directional confidence, illustrating the resilience of weak-form market efficiency over multi-day forecasting horizons.

---

## Dependencies

- **Language:** Python
- **Libraries:** TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib

To run locally:
```bash
git clone [https://github.com/artyom-karapetyan/dl-equity-trend-predictor.git](https://github.com/artyom-karapetyan/dl-equity-trend-predictor.git)
cd dl-equity-trend-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt