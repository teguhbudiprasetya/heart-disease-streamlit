import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from streamlit_option_menu import option_menu
from streamlit_extras.app_logo import add_logo
from process import predicted

def loading():
  with st.spinner('Tunggu Sebentar...'):
    time.sleep(0.3)

add_logo("https://www.google.com/url?sa=i&url=https%3A%2F%2Fkaciicons.tumblr.com%2Fpost%2F712348758156967936%2Fanya-forger-icons&psig=AOvVaw3o0EjeeRR8GmXa74iXi6Kx&ust=1696884152498000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCNCTspuo54EDFQAAAAAdAAAAABAX")
st.markdown("# Implementasi")
selected = option_menu(
    menu_title  = "Proyek Sains Data",
    options     = ["Dataset","Prediksi"],
    orientation = "horizontal",
)


dataset = pd.read_csv('data/dataset_SMOTE.csv')

# Save dataset
# joblib.dump(dataset, 'model/dataset.sav')
if (selected == "Dataset"):
    loading()
    st.success(f"Jumlah Data : {dataset.shape[0]} Data, dan Jumlah Fitur : {dataset.shape[1]-1} Fitur")
    dataframe, keterangan = st.tabs(['Datset', 'Keterangan'])
    with dataframe:
        st.write(dataset)

    with keterangan:
        st.text("""
             Column:
             - Mean: Rata-rata freqs
             - Std: Standar devisiasi freqs
             - Max: Nilai terbesar dari freqs
             - min: Nilai terkecil dari freqs
             - median: Nilai tengah freqs
             - modus: Nilai paling sering muncul dari freqs
             - skew: Kecondongan freqs
             - kurt: Distribusi freqs
             - q1: Titik potong 25 
             - q2: Titik potong 75
             - iqr: Rentang akar kuartil
             - zcr mean: Mean dari ZCR
             - zcr median: Median dari ZCR
             - zcr std: Standar devisiasi dari ZCR
             - zcr kurt: Kurtosis dari ZCR
             - zcr skew: Kecondongan dari ZCR
             - rmse mean: Mean dari RMSE
             - rmse median: Median dari RMSE
             - rmse std: Standar devisiasi dari RMSE
             - rmse kurt: Kurtosis dari RMSE
             - rmse skew: Kecondongan dari RMSE

             Class
             > Sad
             > Fear
             > Happy
             > Angry
             > Neutral
             > Disgust
             > Surprise
           """)


elif(selected == 'Prediksi'):
  loading()
  predicted.prediksi()

