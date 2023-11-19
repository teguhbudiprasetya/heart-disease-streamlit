import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
import librosa
import scipy

def prediksi():
    knn_zero = joblib.load('model/knn_z-score.sav')
    knn_mmax = joblib.load('model/knn_mmax.sav')

    scalar_zero = joblib.load('model/z-score_scaler.sav')
    scalar_mmax = joblib.load('model/mmax_scaler.sav')

    uploaded_audio = st.file_uploader("Unggah audio yang akan diprediksi", type=['mp3','wav','ogg'])

    if uploaded_audio:
        # Display the uploaded file
        st.audio(uploaded_audio)

        x, sr = librosa.load(uploaded_audio, sr=None)
        freqs = np.fft.fftfreq(x.size)

        mean = np.mean(freqs)
        std = np.std(freqs)
        maxv = np.amax(freqs)
        minv = np.amin(freqs)
        median = np.median(freqs)
        skew = scipy.stats.skew(freqs)
        kurt = scipy.stats.kurtosis(freqs)
        modus = scipy.stats.mode(freqs)[0]
        q1 = np.quantile(freqs, 0.25)
        q3 = np.quantile(freqs, 0.75)
        iqr = scipy.stats.iqr(freqs)

        zcr = librosa.feature.zero_crossing_rate(x)
        zcr_mean = np.mean(zcr)
        zcr_median = np.median(zcr)
        zcr_std = np.std(zcr)
        zcr_kurt = scipy.stats.kurtosis(zcr, axis=1)[0]
        zcr_skew = scipy.stats.skew(zcr, axis=1)[0]

        rmse = librosa.feature.rms(y=x)
        rmse_mean = np.mean(rmse)
        rmse_median = np.median(rmse)
        rmse_std = np.std(rmse)
        rmse_kurt = scipy.stats.kurtosis(rmse, axis=1)[0]
        rmse_skew = scipy.stats.skew(rmse, axis=1)[0]

        df = pd.DataFrame({
            'mean': mean,
            'std': std,
            'max': maxv,
            'min': minv,
            'median': median,
            'modus': modus,
            'skew': skew,
            'kurt': kurt,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'zcr_mean': zcr_mean,
            'zcr_median': zcr_median,
            'zcr_std': zcr_std,
            'zcr_kurt': zcr_kurt,
            'zcr_skew': zcr_skew,
            'rmse_mean': rmse_mean,
            'rmse_median': rmse_median,
            'rmse_std': rmse_std,
            'rmse_kurt': rmse_kurt,
            'rmse_skew': rmse_skew,
        })
        st.write(df)

        st.subheader("Z-Score")
        scaler = joblib.load('model/z-score_scaler.pkl')
        knn_zero_model = joblib.load('model/knn_z-score_grid.pkl')

        st.write('Tanpa PCA')
        x_test_zscore = scaler.transform(df)
        y_pred = knn_zero_model.predict(x_test_zscore)
        st.write(y_pred)

        st.write('Dengan PCA')
        knn_zero_PCA_model = joblib.load('model/knn_mmax_grid_PCA.pkl')
        pca = joblib.load('model/PCA_mmax.pkl')
        x_test_zscore_pca = pca.transform(x_test_zscore)
        st.write(x_test_zscore_pca)
        y_pred_pca = knn_zero_PCA_model.predict(x_test_zscore_pca)
        st.write(y_pred)
