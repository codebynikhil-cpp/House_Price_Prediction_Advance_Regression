# Advanced Regression: House Price Prediction 🏡

This project builds an advanced regression model to accurately predict house prices. Developed as part of a Kaggle competition, this solution implements comprehensive data engineering, ensemble learning, and hyperparameter optimization to achieve leading predictive performance.

## 🏆 Project Highlight
This solution achieved a rank of **180 out of 5,000+ teams (Top 4%)** with an impressive **RMSLE score of 0.11899**.

## 🧠 Approach & Technology Stack
The pipeline extensively cleans and transforms the data, engineers new cyclic and boolean features, handles skewness, and mitigates outliers. We utilize an ensemble learning methodology, stacking predictions from multiple optimized models to maximize accuracy.

### Models Used:
- **Lasso Regression**
- **Ridge Regression**
- **Support Vector Regression (SVR)**
- **LightGBM (LGBM)**
- **Gradient Boosting Machine (GBM)**
- **XGBoost (XGB)**

## 📂 Repository Structure
```text
├── Data
│   ├── test.csv        # Testing data
│   └── train.csv       # Training data
├── House-Price-Prediction-Solution.ipynb  # Main Jupyter Notebook
├── requirements.txt    # Required Python dependencies
├── .gitignore          # Git ignore specifications
└── README.md           # Project documentation
```

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/codebynikhil-cpp/House_Price_Prediction_Advance_Regression.git
   cd House_Price_Prediction_Advance_Regression
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   Launch Jupyter Notebook to view the complete EDA, feature engineering, and model training pipeline.
   ```bash
   jupyter notebook House-Price-Prediction-Solution.ipynb
   ```
