This repository contains a predictive model and a web application designed to assess customer churn risk in a financial context. The application uses various machine learning models to predict the likelihood of a customer churning and provides explanations for the prediction, leveraging a large language model (LLM) to generate insights and personalized retention strategies. Additionally, automated email generation offers tailored incentives for at-risk customers.

Features
Customer Churn Prediction: Predicts churn probability based on customer data such as age, location, tenure, and account balance.
Model Explanations: Provides interpretability by generating detailed explanations for each prediction using an LLM.
Personalized Email Generation: Creates personalized emails to encourage loyalty, offering incentives based on individual churn risk factors.
Interactive Web Interface: Built with Streamlit, allowing users to input customer information and view predictions and explanations in real time.
Models Used
The following models were trained using cross-validation to ensure optimal accuracy:

XGBoost (xgb_model.pkl)
Naive Bayes (nb_model.pkl)
Random Forest (rf_model.pkl)
Decision Tree (dt_model.pkl)
SVM (svm_model.pkl)
K-Nearest Neighbors (knn_model.pkl)
Voting Classifier (voting_clf.pkl)
XGBoost (SMOTE) (xgboost_model-smote.pkl)
XGBoost (Feature Engineering) (xgboost_model-fe.pkl)


Code Structure
main.py: Primary Streamlit app script for customer churn prediction and interaction.
utils.py: Contains helper functions for data processing and visualization.
Model Files: Pre-trained models saved as .pkl files for loading into the app.
Dataset: churn.csv - Dataset of customer information used for predictions and insights.
Explanation and Email Generation
The app utilizes an OpenAI-powered LLM to explain predictions based on feature importances and customer summary statistics. Here’s how it works:

Prediction Explanation: The LLM analyzes customer attributes and outputs a natural-language explanation based on their churn probability, feature importances, and summary statistics for churned and non-churned customers.
Email Generation: If a customer has a high churn probability, the LLM generates a retention email with specific incentives tailored to mitigate risk factors identified in their profile.
