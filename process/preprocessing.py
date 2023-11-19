import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data/dataset.csv')

def splitting():
    st.success('Data dibagi menjadi 80:20, 80% data training dan 20% data testing')
    train_tabs, test_tabs = st.tabs(['Traning', 'Testing'])
    x_train, x_test, y_train, y_test = train_test_split(dataset.drop(columns=['label']), dataset['label'], test_size=0.2, random_state=42)
    # joblib.dump(df_gabung, 'model/df_train_pre.sav')
    with train_tabs:
        st.write(f"Jumlah Data Training : {x_train.shape[0]} Data")
        st.write(x_train)
    with test_tabs:
        st.write(f"Jumlah Data Testing : {x_test.shape[0]} Data")
        st.write(x_test)

    joblib.dump(x_train, 'model/df_train_split.sav')
    joblib.dump(x_test, 'model/df_test_split.sav')
    joblib.dump(y_train, 'model/y_train_split.sav')
    joblib.dump(y_test, 'model/y_test_split.sav')
def z_score():
    x_train = joblib.load('model/df_train_split.sav')
    st.write('Data Awal Sebelum di lakukan Normalisasi')
    st.dataframe(x_train)

    st.write('Data setelah dilakukan Preprocessing menggunakan Z-Score Scaler')
    z_scaler = joblib.load('model/z-score_scaler.pkl')
    train_scaled_z_score = z_scaler.transform(x_train)
    st.write(train_scaled_z_score)

    # Save Scaled
    joblib.dump(train_scaled_z_score, 'model/df_train_z-score.sav')
    # joblib.dump(z_scaler, 'model/z-score_scaler.sav')

def minMax():
    x_train = joblib.load('model/df_train_split.sav')
    st.write('Data Awal Sebelum di lakukan Normalisasi')
    st.dataframe(x_train)

    st.write('Data setelah dilakukan Preprocessing menggunakan Min-Max Scaler')
    mmax_scaler = joblib.load('model/mmax_scaler.pkl')
    train_scaled_mmax = mmax_scaler.transform(x_train)
    st.write(train_scaled_mmax)

    # Save Scaled
    joblib.dump(train_scaled_mmax, 'model/df_train_mmax.sav')
    # joblib.dump(mmax_scaler, 'model/mmax_scaler.sav')

