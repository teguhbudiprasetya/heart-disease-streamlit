import pandas as pd
import numpy as np
import os
import pickle
import joblib
import streamlit as st
import scipy
import sklearn

def prediksi():

    rf_model = joblib.load('model/random_forest_SMOTE.pkl')
    xg_model = joblib.load('model/XGBoost_SMOTE.pkl')
    nn_model = joblib.load('model/Neural_Network_SMOTE.pkl')
    sv_model = joblib.load('model/SVM_SMOTE.pkl')
    lr_model = joblib.load('model/Logistic_SMOTE.pkl')
    nb_model = joblib.load('model/Naive_SMOTE.pkl')
    minmax = joblib.load('model/scaler_mmax.pkl')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=63)
        sex = st.selectbox('Sex', ['Male', 'Female'], index=0)
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'], index=0)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=1, value=145)
        chol = st.number_input('Cholesterol (mg/dL)', min_value=1, value=230)

    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'], index=1)
        restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'], index=2)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=1, value=150)
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'], index=1)
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, value=2.3)

    with col3:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'], index=2)
        ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, value=0)
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversable Defect'], index=1)

    # Button to trigger prediction
    if st.button('Predict'):
        # Use the entered values for further processing or prediction
        if sex == 'Male':
            sex = 1
        else:
            sex = 0

        if cp == 'Typical Angina':
            cp = 1
        elif cp == 'Atypical Angina':
            cp = 2
        elif cp == 'Non-anginal Pain':
            cp = 3
        else:
            cp = 4

        if fbs == 'True':
            fbs = 1
        else:
            fbs = 0

        if restecg == 'Normal':
            restecg = 0
        elif restecg == 'ST-T Wave Abnormality':
            restecg = 1
        else:
            restecg = 2

        if exang == 'No':
            exang = 0
        else:
            exang = 1

        if slope == 'Upsloping':
            slope = 1
        elif slope == 'Flat':
            slope = 2
        else:
            slope = 3

        if thal == 'Normal':
            thal = 3
        elif thal == 'Fixed Defect':
            thal = 6
        else:
            thal = 7

        st.write("Perform prediction or processing with the entered values.")

        x_test = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        print(x_test)
        scaled_test = minmax.transform(x_test)
        print(scaled_test)
        st.success(f'Random Forest  :{rf_model.predict(scaled_test)[0]}')
        # st.success(f'XGBoost        :{xg_model.predict(scaled_test)[0]}')
        # st.success(f'Neural Network :{nn_model.predict(scaled_test)[0]}')
        # st.success(f'Logistic Regression :{lr_model.predict(scaled_test)[0]}')
        # st.success(f'SVM :{sv_model.predict(scaled_test)[0]}')
        # st.success(f'Naive Bayes :{nb_model.predict(scaled_test)[0]}')





