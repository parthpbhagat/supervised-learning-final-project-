# Ames Housing Price Prediction & Exploratory Data Analysis (EDA)

This repository contains a comprehensive Machine Learning workflow for predicting residential home prices in Ames, Iowa. The project covers end-to-end steps: Exploratory Data Analysis (EDA), advanced data preprocessing, feature engineering, regression modeling (Linear, Ridge, Lasso, Random Forest, and XGBoost), hyperparameter tuning, and model interpretation using SHAP (SHapley Additive exPlanations) values.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Workflow & Pipeline](#-workflow--pipeline)
- [Key Features & Engineering](#-key-features--engineering)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Key Insights & Feature Importance](#-key-insights--feature-importance)
- [Installation & Usage](#-installation--usage)

---

## 🔍 Project Overview

The objective of this project is to build an accurate and interpretable regression model to predict the target variable `SalePrice`. The dataset used is the Ames Housing dataset, which consists of 79 explanatory features describing various aspects of residential homes.

---

## ⚙️ Workflow & Pipeline

The pipeline is implemented sequentially in the Jupyter notebook `Ames_Housing_EDA.ipynb`:

1. **Exploratory Data Analysis (EDA)**
   - Analysis of target variable distribution (`SalePrice`) and applying log-transformation (`log1p`) to fix positive skewness.
   - Univariate analysis on categorical and skewed numerical features.
   - Bivariate analysis to discover correlations and relationships (e.g., Boxplots of `OverallQual` vs `SalePrice`).
   - Identifying and flagging extreme outliers (e.g., houses with `GrLivArea > 4000` sq. ft. sold for very low prices).

2. **Data Preprocessing**
   - **Handling Missing Values**: Custom imputation logic (e.g., imputing categorical missing values with `'None'` or mode, and numerical ones with `0` or median).
   - **Outlier Removal**: Filtering out verified leverage points that harm linear models.
   - **Feature Scaling & Encoding**: Using `ColumnTransformer` with `StandardScaler` for numerical columns and `OneHotEncoder(handle_unknown='ignore')` for categorical columns.

3. **Feature Engineering**
   - Combining spaces to create new interactive features like `TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`.
   - Age calculation at the time of sale.

4. **Modeling & Hyperparameter Tuning**
   - Evaluated 5 baseline algorithms: Linear Regression, Ridge Regression (L2), Lasso Regression (L1), Random Forest, and XGBoost.
   - Tuned hyperparameters using `RandomizedSearchCV` with 5-fold cross-validation.

5. **Model Interpretation**
   - Feature weights/coefficients analysis for Lasso.
   - SHAP Beeswarm and Waterfall plots to explain global feature contributions and individual predictions.

---

## 📊 Model Evaluation & Results

The models were evaluated on the test set using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$).

| Model | Test RMSE | Test MAE | Test $R^2$ Score | 5-Fold CV RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Lasso Regression (L1)** | **$20,089.42** | **$14,731.01** | **0.9269** | **$23,170.12** |
| **XGBoost Regressor** | $20,406.50 | $14,616.10 | 0.9246 | $23,557.54 |
| **Ridge Regression (L2)** | $20,492.23 | $14,682.58 | 0.9240 | $22,585.97 |
| **Linear Regression (OLS)** | $22,610.59 | $15,817.66 | 0.9074 | $26,154.70 |
| **Random Forest** | $23,415.72 | $16,428.29 | 0.9007 | $27,453.52 |

### 💡 Key Takeaways:
- **Lasso Regression** performed best overall, achieving an **$R^2$ of 92.69%**. It successfully regularized the feature space, setting coefficients of **121 out of 187** features to exactly zero.
- **XGBoost** also performed strongly, with a test RMSE of **$20,406.50**.

---

## 🔑 Key Insights & Feature Importance

Based on Lasso coefficients, the top 5 most impactful features on house prices are:

1. **GrLivArea** (Above grade living area square feet): **+0.1192** 📈
2. **SaleType_New** (New home construction): **+0.1082** 📈
3. **OverallQual** (Overall material and finish quality rating): **+0.0746** 📈
4. **Functional_Typ** (Typical home functionality): **+0.0629** 📈
5. **HouseAge** (Age of the house at sale): **-0.0520** 📉 *(Negative impact: older houses fetch lower prices)*

---

## 🚀 Installation & Usage

### Prerequisites
Make sure you have Python 3.8+ installed. You can install all required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

### Running the Project

To execute the notebook from the command line and update the outputs:
```bash
python -m nbconvert --to notebook --execute Ames_Housing_EDA.ipynb --inplace
```

Or run it interactively by opening the notebook in **Jupyter Lab** or **VS Code**:
```bash
jupyter lab
```
And open [Ames_Housing_EDA.ipynb](file:///d:/ml%20final project/Ames_Housing_EDA.ipynb).
