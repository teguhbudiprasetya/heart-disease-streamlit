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
             Fitur:
             - age: umur dalam tahun
             - sex: jenis kelamin
             - cp: chest pain type (tipe penyakit dada)
             - trestbps: tekanan darah (dalam mm Hg)
             - chol: kolestrol (dalam mg/dl)
             - fbs: (gula darah > 120 mg/dl)
             - restecg: hasil kondisi ECG pasien saat sedang istirahat
             - thalach: detak jantung maksimal yang diraih permenit
             - exang: nyeri dada setelah berolahraga
             - oldpeak:  penurunan segment ST pada elektrokardiogram (EKG) yang terjadi selama tes latihan fisik
             - slope: tkemiringan segmen ST selama uji latihan fisik pada elektrokardiogram (EKG)
             - ca: jumlah pembuluh darah utama yang ditemukan dalam gambaran angiografi koroner yang diwarnai dengan fluoroskopi (0-3)
             - thal: stress test atau tes latihan stres thallium, mengevaluasi aliran darah ke otot jantung selama latihan fisik dan pada istirahat


             Kelas
             > Value 0: Tidak ada penyakit jantung
             > Value 1: Penyakit jantung ringan
             > Value 2: Penyakit jantung sedang
             > Value 3: Penyakit jantung signifikan
             > Value 4: Penyakit jantung parah
           """)


elif(selected == 'Prediksi'):
  loading()
  predicted.prediksi()

