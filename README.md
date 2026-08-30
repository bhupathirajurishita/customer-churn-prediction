 📊 Customer Churn Prediction

A machine learning classification project that predicts customer churn using customer-related data and compares multiple classification models.

 📌 Project Overview

Customer churn prediction helps identify customers who may stop using a service.

In this project, different machine learning classification models are trained, evaluated, and compared. Hyperparameter tuning is also performed on the Random Forest model to improve its performance.

 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Google Colab

 📊 Dataset

The dataset contains customer-related information along with a `Churn` target variable.

The project uses the available customer features to predict whether a customer will churn.

 ⚙️ Project Workflow

1. Load the dataset.
2. Perform data preprocessing.
3. Check for missing values and duplicate records.
4. Separate features and target variable.
5. Split the data into training and testing sets.
6. Train a Decision Tree classifier.
7. Train a Random Forest classifier.
8. Train an XGBoost classifier.
9. Compare the initial model performance.
10. Perform hyperparameter tuning using `GridSearchCV`.
11. Evaluate the tuned Random Forest model.
12. Compare the tuned model with the initial models.
13. Select the best-performing model.
14. Make a sample prediction.

 🤖 Models Used

Decision Tree

Used as a baseline classification model for predicting customer churn.

Random Forest

A tree-based ensemble classification algorithm used to improve prediction performance.

XGBoost

A gradient boosting algorithm used for classification.

Hyperparameter Tuning

`GridSearchCV` was used to search for better Random Forest hyperparameters.

The best parameters found were:

```text
max_depth = 10
min_samples_split = 2
n_estimators = 200
