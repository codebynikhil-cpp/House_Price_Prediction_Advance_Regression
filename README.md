# Advanced Regression: House Price Prediction 🏡

This project solves the Kaggle House Prices prediction problem using advanced regression techniques and ensemble learning.
The goal is to accurately predict the sale price of homes in Ames, Iowa using structured dataset features such as lot size, year built, neighborhood, and house condition.

## 🚩 What This Project Does

- Loads and inspects the house price training and test datasets.
- Analyzes relationships between features and SalePrice using visualization and statistics.
- Cleans the data by handling missing values, correcting feature types, and engineering new variables.
- Combines train and test sets for consistent preprocessing.
- Encodes categorical features, scales numeric features, and removes outliers.
- Tunes multiple regression models and combines them in a stacked ensemble.
- Produces predictions for the test set that can be submitted to Kaggle.

## 🧠 Main Modeling Approach

The notebook builds a hybrid regression pipeline with the following key components:

1. **Exploratory Data Analysis (EDA)**
   - Identify outliers and feature distributions.
   - Evaluate missing values and numeric/skewed data behavior.

2. **Feature Engineering**
   - Create new features from existing columns.
   - Convert categorical text values into numeric representations.
   - Apply transformations to reduce skewness.

3. **Preprocessing**
   - Merge training and test data for consistent transformations.
   - Impute missing values using suitable strategies.
   - One-hot encode categorical variables with `pd.get_dummies()`.
   - Scale numeric features using `RobustScaler` to reduce outlier impact.
   - Remove extreme outliers identified by residual analysis.

4. **Model Training & Tuning**
   - Use cross-validation and `RandomizedSearchCV` for hyperparameter tuning.
   - Train multiple base regressors and compare their performance.

5. **Ensemble Stacking**
   - Stack multiple tuned regressors with a Ridge meta-learner.
   - This improves final prediction stability and accuracy.

## 📦 Models Used

The notebook explores and uses the following regression models:

- **Lasso Regression** — regularized linear model that reduces overfitting.
- **Ridge Regression** — linear model with L2 regularization.
- **Support Vector Regression (SVR)** — kernel-based regression for complex relationships.
- **LightGBM Regressor** — fast gradient boosting tree model.
- **Gradient Boosting Regressor** — tree-based boosting with strong predictive power.
- **XGBoost Regressor** — optimized gradient boosting implementation.
- **CatBoost Regressor** — gradient boosting that handles categorical features well.
- **Random Forest Regressor** — ensemble of decision trees.

The final prediction uses a **stacking ensemble** that combines multiple models and learns the best blend.

## 📈 Evaluation

- The main evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)**.
- The project achieves strong results with a final notebook score around **0.11899 RMSLE**.
- This performance ranked the solution in the **top 4%** of competition submissions.

## 📂 Repository Structure

```text
├── Data
│   ├── test.csv        # Hold-out dataset for final predictions
│   └── train.csv       # Training dataset with SalePrice target
├── House-Price-Prediction-Solution.ipynb  # Full analysis and training notebook
├── requirements.txt    # Python package dependencies
├── .gitignore          # Files and folders excluded from Git
└── README.md           # Project documentation
```

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook House-Price-Prediction-Solution.ipynb
   ```

3. Run cells in order to reproduce the full workflow:
   - data loading
   - preprocessing
   - feature engineering
   - encoding and scaling
   - model training and stacking
   - prediction generation

## 🎓 What to Tell Your Teacher

When presenting this project, explain that you:

- Solved a supervised machine learning regression problem.
- Used **feature engineering** and **data cleaning** to improve model quality.
- Applied both **linear** and **tree-based** regression models.
- Tuned hyperparameters using **cross-validation**.
- Built a **stacked ensemble**, which is a model of models.
- Evaluated the solution using **RMSLE**, which is appropriate for price predictions.
- Produced a strong Kaggle result and documented the full process in a notebook.

## 💡 Notes

- The notebook is the main deliverable; it contains charts, code comments, and explanations.
- If you want to test a different model, add it to the ensemble section and compare results.
- The `requirements.txt` file includes the packages needed to run the notebook.
