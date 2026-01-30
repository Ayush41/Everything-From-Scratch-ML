# 📄 Research Paper: Comparative Analysis of ML & Statistical Models for CO2 Forecasting

**Project Status:** 🟢 Active / Experimentation Phase  
**Research Lead:** [Your Name]  
**Supervisor/Mentor:** [Professor's Name]  
**Topic:** Time-Series Forecasting of Global Greenhouse Gas Emissions (1970–2030)

---

## 📌 1. Abstract & Problem Statement
This research aims to develop, validate, and compare predictive models for forecasting Annual CO2 Emissions across key industrial nations (**India, USA, China, UK, Germany**). 

The core objective is to analyze the efficacy of **Machine Learning (ML)** approaches versus traditional **Statistical (ARIMA)** methods in handling long-term environmental data. The study focuses on a **Univariate Time-Series approach** to avoid "error propagation" from external covariates, ensuring forecasts are driven strictly by historical emission trends and cyclical patterns.

### **Key Objectives**
1.  **Analyze Historical Trends:** Deep dive into the EDGAR dataset (1970–2018).
2.  **Model Validation:** rigorously test models on unseen data (2019–2025).
3.  **Future Forecasting:** Project emission trajectories for the next 5 years (2025–2030).
4.  **Comparative Study:** Benchmarking Linear vs. Non-Linear algorithms.

---

## 📂 2. Dataset Information
**Source:** EDGAR (Emissions Database for Global Atmospheric Research)  
**Temporal Coverage:** 1970 – 2025  

### **Selected Features (Input Files)**
We are focusing on three critical dimensions of carbon emissions:
* `fossil_co2_total_by_country.csv` - Absolute emission volume (Mt).
* `co2_per_gdp_by_country.csv` - Economic emission intensity.
* `co2_per_capita_by_country.csv` - Individual footprint metrics.

> **Note:** Sector-specific data (`fossil_co2_by_sector.csv`) and metadata (`info.csv`, `citations.csv`) are excluded to maintain a focused univariate analysis scope.

---

## 🔬 3. Methodology & Experimentation
We employ a **Multi-Model Approach** to determine the most robust forecasting technique for different economic lifecycles (Developing vs. Developed nations).

### **A. Statistical Approach (Baseline)**
* **Model:** **ARIMA / SARIMA** (AutoRegressive Integrated Moving Average).
* **Role:** Acts as the statistical baseline.
* **Strength:** Excellent for capturing linear trends and seasonality without complex feature engineering.

### **B. Machine Learning Approach (challengers)**
* **Models:** * **Linear Regression:** To establish a linear baseline.
    * **Random Forest Regressor:** To capture non-linear complexities and regime shifts in developing economies.
* **Feature Engineering (The "Lag" Strategy):** Since ML models are not inherently temporal, we transform the time-series problem into a supervised learning problem using **Lag Features**:
    * *Input:* $X = [t-1, t-2, t-3]$ (Emissions from previous 3 years)
    * *Target:* $Y = [t]$ (Emission for current year)

### **C. Training & Validation Split**
To prevent "Look-ahead Bias" (Temporal Leakage), we strictly split data by time, not randomization.
* **Training Set:** 1970 – 2018 (Pattern Recognition)
* **Testing Set:** 2019 – 2025 (Model Validation against known reality)
* **Forecasting Horizon:** 2026 – 2030 (Pure Future Projection)

---

## 📊 4. Evaluation Metrics
We focus on regression metrics rather than classification accuracy.

| Metric | Definition | Why we use it |
| :--- | :--- | :--- |
| **MAE** | Mean Absolute Error | Measures average deviation in Megatons. Easy to interpret. |
| **MAPE** | Mean Absolute Percentage Error | Expresses error as a % (e.g., "Off by 5%"). Crucial for comparing countries with vastly different emission scales (e.g., USA vs. UK). |
| **RMSE** | Root Mean Squared Error | Penalizes larger errors more heavily (good for spotting outliers). |
| **R²** | Coefficient of Determination | Indicates how well the model explains the variance in the data. |

---

## 🛠️ 5. Tech Stack & Dependencies
* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy` (Data Manipulation)
* **Visualization:** `matplotlib`, `seaborn` (Trend & Error Plots)
* **Modeling:** `statsmodels` (ARIMA), `scikit-learn` (Random Forest, Linear Regression, Metrics)

---

## 🔮 6. Expected Outcomes
1.  Accurate, data-driven forecasts for 2030 emissions to aid in policy analysis.
