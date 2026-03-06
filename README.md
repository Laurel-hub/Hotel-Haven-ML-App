AI-Powered Booking Cancellation Risk Dashboard

A machine learning web application that predicts the probability of a hotel booking being cancelled and provides operational insights for revenue and booking management.

This project demonstrates an end-to-end data science workflow including data preprocessing, machine learning modelling, and interactive deployment using Streamlit.

The application allows users to input booking information and receive a real-time cancellation risk prediction powered by a Random Forest classifier.
Live Application

Access the deployed dashboard here:

Streamlit App:
https://hotel-haven-ml-app-cvkrenkwdcneue77mkd8xy.streamlit.app/
Project Overview

Booking cancellations present a significant operational and financial challenge for hotels. Accurately predicting cancellation risk allows hospitality providers to optimise pricing strategies, improve overbooking policies, and enhance revenue management.

This project builds a predictive model trained on historical booking data and deploys it as an interactive dashboard for real-time risk estimation.

Users can enter booking details such as:

Lead time

Market segment

Room type

Meal plan

Number of guests

Length of stay

Reservation date

The model then estimates the probability that the booking will be cancelled and displays the risk percentage.

Machine Learning Pipeline

The application follows a typical production-style machine learning workflow:

Data ingestion and preprocessing

Feature cleaning and selection

Model training using Random Forest

Feature alignment between training and prediction

Deployment via Streamlit

The trained model predicts cancellation probability using the predict_proba() method from scikit-learn.

Technologies Used

Python
Pandas
NumPy
Scikit-Learn
Streamlit
Git & GitHub

ile Description

app.py
Main Streamlit application containing data preprocessing, model training, and the interactive prediction interface.

booking.csv
Dataset used to train the cancellation prediction model.

feature_columns.json
Stores the exact feature column structure used during training to ensure consistent predictions during deployment.

requirements.txt
List of Python dependencies required to run the application.

Model Details

Model Type: Random Forest Classifier

Random Forest was selected due to its ability to handle:

non-linear relationships

mixed feature types

high dimensional feature spaces

The model outputs cancellation probabilities which are converted to a risk percentage displayed in the dashboard.

Example Prediction Workflow

User enters booking details

Inputs are converted into a structured feature dataframe

Features are aligned with the training dataset

The model predicts cancellation probability

Risk percentage is displayed in the dashboard

Learning Outcomes

This project demonstrates practical experience in:

machine learning model deployment

feature engineering

handling feature mismatch during prediction

building interactive data applications

deploying ML models to the cloud

Future Improvements

Potential enhancements include:

adding model evaluation metrics

incorporating feature importance visualisations

adding SHAP explainability for predictions

storing the trained model as a serialized pipeline

integrating a database for real booking data

Author

Oghenevurie Lauretta
MSc Artificial Intelligence
University of Essex Online




