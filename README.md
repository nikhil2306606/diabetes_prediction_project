Diabetes Prediction using XGBoost – Pima Dataset

Project Overview

This project is a real-world machine learning portfolio project focused on predicting diabetes using patient health data from the Pima Indians Diabetes dataset. The project demonstrates end-to-end data analysis, statistical exploration, model building, evaluation, and prediction.

The objective of this project is to simulate a real-world data science workflow, from exploratory data analysis (EDA) to building a high-performing predictive model. 

The project includes:
* Data cleaning and preprocessing
* Statistical analysis including summary statistics, correlation analysis, t-tests, and chi-square tests
* Data visualization using plots, graphs, and heatmaps
* Model selection, training, and evaluation using XGBoost
* Feature importance analysis

Dataset Description:
The Pima Indians Diabetes dataset contains medical data for female patients of Pima Indian heritage. Each row represents an individual patient with several health-related attributes.
Key Columns:
* Pregnancies: Number of pregnancies
* Glucose: Plasma glucose concentration
* BloodPressure: Diastolic blood pressure (mm Hg)
* SkinThickness: Triceps skinfold thickness (mm)
* Insulin: 2-Hour serum insulin (mu U/ml)
* BMI: Body mass index (weight in kg/(height in m)^2)
* DiabetesPedigreeFunction: Diabetes pedigree function (family history)
* Age: Patient age (years)
* Outcome: Class variable (0 = Non-Diabetic, 1 = Diabetic)

Statistical Analysis Performed
* Summary statistics of all attributes
* Distribution plots for numerical features
* Correlation matrix and heatmap visualization
* T-tests to compare means between diabetic and non-diabetic groups
* Chi-square tests for categorical feature associations

Machine Learning Model – XGBoost
* XGBoost (Extreme Gradient Boosting) was selected for its high performance and ability to handle feature interactions effectively
* Features were scaled using StandardScaler to improve model convergence
* Model evaluation included metrics like Accuracy, Sensitivity (Recall), Specificity, and Confusion Matrix
* Feature importance analysis to identify the most influential factors in diabetes prediction

Model Performance

* Accuracy: 0.8839 – The model correctly predicts diabetes status for approximately 88% of patients.
* Sensitivity (Recall): 0.8704 – The model correctly identifies about 87% of actual diabetic patients (true positives).
* Specificity: 0.8911 – The model correctly identifies about 89% of non-diabetic patients (true negatives).

Note: These results are based on the current dataset and model. Predictions may not be perfectly accurate for all patients or real-world scenarios, and there may be errors. This model is intended for educational and analytical purposes only.

Usage

* The trained XGBoost model and scaler are saved as `xgb_model.pkl` and `scaler.pkl`
* Users can load the model to predict diabetes for new patients
* The system can append new patient records to the dataset for future analysis


---------------------- Author -------------------------
Nikhil Pereira
Aspiring Data Scientist| BSc Statistics & Computer Science

Learning through real-world projects and building a strong data portfolio
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/nikhilpereira23/)
