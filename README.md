# Predictive Maintenance of Wind Turbine Systems using AI

This repository contains the code and analysis for predicting maintenance needs in wind turbine systems using Artificial Intelligence. The project leverages advanced data analytics and machine learning techniques to enhance the reliability and efficiency of wind energy systems through predictive maintenance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Approach](#approach)
- [Notebooks](#notebooks)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [Requirements](#requirements)


## Project Overview
Predictive maintenance is crucial for the wind energy sector to minimize downtime and maximize operational efficiency. This project uses AI techniques to predict potential failures and maintenance requirements, ensuring the reliability and safety of wind turbine systems.

## Dataset
The project utilizes simulated wind turbine data, including:
- Operational parameters like wind speed, power output, and reactive power.
- Event logs and failure descriptions for anomaly detection and failure classification.

## Objectives
- Predict potential failures in wind turbine systems using AI.
- Classify different types of failures for targeted maintenance actions.
- Estimate Remaining Useful Life (RUL) for critical components.
- Enhance reactive power management for improved energy efficiency.

## Approach
1. **Data Preprocessing:** Cleaning and transforming data for better model performance.
2. **Exploratory Data Analysis (EDA):** Understanding data distributions and relationships.
3. **Feature Engineering:** Creating new features to improve model accuracy.
4. **Model Development:** Using machine learning models for anomaly detection, failure classification, and RUL prediction.
5. **Evaluation and Interpretation:** Analyzing model performance and deriving actionable insights.

## Notebooks
The project is organized into the following Jupyter Notebooks:
- [Basic EDA](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/Basic_EDA.ipynb): Initial data exploration and visualization.
- [Anomaly Detection](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/Anomaly_detection.ipynb): Identifying unusual patterns and anomalies.
- [Correlation Analysis](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/Correlation_analysis.ipynb): Investigating feature relationships.
- [Data Preprocessing](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/Dp_wind%20(3).ipynb): Data cleaning and transformation.
- [Failure Types](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/Failure_types.ipynb): Classification of different failure types.
- [RUL Prediction](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/RUL_thanu.ipynb): Estimating the Remaining Useful Life of components.
- [Reactive Power Management](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/Reactive_Power_Management.ipynb): Enhancing energy efficiency through reactive power management.
- [Feature Selection](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/feature_selection.ipynb): Selecting the most relevant features for modeling.
- [Local Event Description](https://github.com/thanusri1601/Predictive-Maintenance-Of-Wind-Turbine-Systems-using-AI/blob/main/local_event_description.ipynb): Analysis of local events impacting turbine performance.

## Modeling Techniques
The project explores the following machine learning and deep learning models:
- Anomaly Detection using Isolation Forest and Autoencoders.
- Failure Classification using Random Forest and XGBoost.
- RUL Prediction using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks.

## Results
- Achieved high accuracy in predicting turbine failures and estimating RUL.
- Identified key features influencing turbine performance and failure types.
- Provided actionable insights for predictive maintenance and operational efficiency.

## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - keras
  - matplotlib
  - seaborn
  - xgboost

