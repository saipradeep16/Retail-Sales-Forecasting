# 🧠 An Enhanced Machine Learning Model for Prediction of Retail Sales

This repository presents a comprehensive approach to predicting retail sales using both **structured** and **time-series** datasets. The project implements classical machine learning algorithms and advanced deep learning models—including a hybrid model—to demonstrate the effectiveness of different techniques across different data contexts.

---

## 📦 Datasets

### 🛒 BigMart Sales Dataset
- **Type:** Structured (non-temporal)
- **Rows:** 8,523
- **Features:** Product-level and Outlet-level metadata
- **Target:** `Item_Outlet_Sales`
- **Source:** [Kaggle](https://www.kaggle.com/datasets/lokeshmendake/big-mart-sales-dataset)

### 🏬 Walmart Sales Dataset
- **Type:** Time-series
- **Period:** 2010–2012 (weekly data)
- **Stores:** 45
- **Features:** `Weekly_Sales`, `Store`, `Date`, `Holiday_Flag`, `CPI`, `Fuel_Price`, `Unemployment`
- **Source:** [Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

---

## 🔍 Methodologies

### BigMart Dataset (Structured ML)
Each model was trained on preprocessed and feature-engineered data using 80/20 train-test split and evaluated via MAE, RMSE, and R².

Models implemented:
- **Linear Regression**
- **Polynomial Regression**
- **Decision Tree Regression**
- **Random Forest Regression**
- **XGBoost Regression**

> Feature engineering included imputation, one-hot encoding, scaling, and outlier treatment using IQR.

### Walmart Dataset (Time-Series Deep Learning)
Two models were built to capture the temporal patterns:
- **LSTM**: Sequential model using 52-week lag windows with batch normalization and dropout.
- **Hybrid LSTM + XGBoost**: Combines LSTM predictions with XGBoost to model residual patterns and feature interactions.

> Time-series features include: Lag values (1–8 weeks), rolling means, holiday flags, CPI, and fuel prices. All features normalized using MinMaxScaler.

---

## 📈 Evaluation Metrics

| Model                  | Dataset     | MAE    | RMSE   | R² Score |
|------------------------|-------------|--------|--------|----------|
| Linear Regression      | BigMart     | 856.34 | 1132.57| 0.48     |
| Polynomial Regression  | BigMart     | 791.27 | 1045.29| 0.58     |
| Decision Tree          | BigMart     | 622.45 | 882.13 | 0.69     |
| Random Forest          | BigMart     | 578.92 | 1084.20| 0.71     |
| XGBoost                | BigMart     | 738.90 | 1015.70| 0.75     |
| LSTM                   | Walmart     | 505.10 | 671.20 | 0.91     |
| Hybrid LSTM + XGBoost  | Walmart     | 386.80 | 528.40 | **0.96** |

---

## 📌 Conclusion

This project confirms that:

- **Classical ML models** are effective for structured data without temporal dependencies. Among them, **XGBoost** showed the best performance.
- **Time-aware deep learning models** like **LSTM** significantly outperform ML models when the dataset contains sequential dependencies.
- A **hybrid architecture (LSTM + XGBoost)** further enhances performance by leveraging both long-term dependencies and structured interactions.
- Model selection should be **data-centric**, not algorithm-centric. Choosing the right architecture depends heavily on the structure and nature of the input dataset.

---

## 🗂️ Project Structure

```bash
├── data/
│   ├── Big_Mart_Sales_Data.csv
│   ├── Walmart.csv
├── notebooks/
│   ├── BigMart_Sales_ML_Models
│   ├── LSTM_Final.ipynb
│   ├── LSTM_and_XGBoost.ipynb
├── Report.pdf
├── README.md
