# Heart Disease Prediction Web App


## Overview

This web application predicts the likelihood of heart disease based on user-entered health data. It utilizes an XGBoost model trained on historical data from Kaggle, offering users real-time predictions and insights into their cardiovascular health. The project is end-to-end, featuring data storage using Hopsworks store, continuous training with GitHub Actions, and a batch inference pipeline for efficient model updates.

## Deployed Model

The predictive model used in this application is continuously updated and deployed [here](https://scalable-heart-prediction.streamlit.app/)

## Dataset

Dataset that has been used to train the model for this project can be found [here](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)

## Features

Predictive Analytics: Predicts heart disease risk using an XGBoost machine learning model.

User Data Storage: Stores user health data securely for ongoing model training and improvement.

Automated Model Updates: GitHub Actions automate model training and updates using batch inference pipelines.

Interactive Interface: User-friendly web interface for inputting health data and viewing prediction results.

## Technologies Used:

Python

XGBoost

GitHub Actions for CI/CD

Hopsworks store for model and feature storage

Streamlit for UI and deoployment

## Disclaimer
Please note that the predictions made by this application are based on machine learning models trained on historical data. They should not be considered as medical advice. Always consult a healthcare professional for any medical concerns or decisions.