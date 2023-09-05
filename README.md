# Telecom Customer Churn Analysis and Prediction

## Overview
This Python script is designed to analyze and predict customer churn in a telecom dataset. Customer churn refers to the phenomenon where customers leave a service or company, and it's a critical problem for businesses to address. In this script, we perform the following tasks:

## Data Preprocessing:

- Import necessary packages for data analysis and visualization.
- Load the dataset from an Excel file into a Pandas DataFrame.
- Examine the dataset's structure and data types.
- Separate the features into numerical and categorical data for exploratory data analysis (EDA).

## Exploratory Data Analysis (EDA):

- Visualize the distribution of the target variable (Churn) to understand class imbalance.
- Examine numerical features for their probability distributions and check for outliers.
- Handle missing values by replacing NaN values with the mean value in the 'Monthly_Bill' column.
- Encode categorical features into numerical values using Label Encoding.

## Feature Selection:

- Use SelectKBest to select the top k most important features based on their correlation with the target variable (Churn).

## Model Building:

- Split the dataset into training and testing sets.
- Address class imbalance using the SMOTEENN technique, which combines oversampling and cleaning.
- Train several machine learning models, including Logistic Regression, Random Forest, Decision Tree, and Gradient Boosting, on the oversampled dataset.
- Evaluate the models using accuracy, confusion matrix, and classification report metrics.

## Hyperparameter Tuning:

- Optimize the hyperparameters of the Gradient Boosting Classifier using Randomized Search CV to improve model performance.

## Cross-Validation:

- Perform 5-fold cross-validation to assess the model's performance more robustly.

## Hyperparameter Tuning (Optional):

- Use Grid Search to find the best hyperparameters for the Gradient Boosting Classifier.

## Model Saving:

- Save the trained Gradient Boosting Classifier model using the pickle library for future use.

## Prediction for New Data:

- Provide a mechanism to make predictions for new customer data.
- Encode the input data and use the trained model to predict whether a customer is likely to churn or continue.
- Display the prediction and confidence level.

## How to Use
1. Install the required Python packages specified at the beginning of the script (NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, imbalanced-learn).

2. Make sure to have the telecom customer dataset in an Excel file named 'customer_churn_large_dataset.xlsx' in the same directory as the script.

3. Execute the script in a Python environment (e.g., Jupyter Notebook).

4. After model training and tuning, you can use the provided mechanism to make predictions for new customer data by inputting the customer's information (e.g., Location, Subscription Length, Monthly Bill, Total Usage, Churn status).

## Conclusion
This script offers a comprehensive analysis of telecom customer churn data, including data preprocessing, feature selection, model building, and prediction. It aims to help businesses identify and retain customers who are at risk of churning. By implementing machine learning techniques and handling class imbalance, the script provides insights into improving customer retention strategies.
