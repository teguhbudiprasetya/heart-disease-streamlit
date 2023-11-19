# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
#
#
# y_train = joblib.load('model/y_train_split.sav')
# y_test = joblib.load('model/y_test_split.sav')
# x_test = joblib.load('model/df_test_split.sav')
#
#
# def knn_zero():
#     st.subheader('Z-Score')
#     # Preprocessing Z-Score
#     x_train_zscore = joblib.load('model/df_train_z-score.sav')
#     scaler = joblib.load('model/z-score_scaler.pkl')
#     x_test_zscore = scaler.transform(x_test)
#
#     knn_zero_model = joblib.load('model/knn_z-score_grid.pkl')
#     st.write('Tanpa PCA')
#     st.success(f'Parameter terbaik K: 5')
#
#     # st.success(f'Parameter terbaik: \n\nK :{grid_search.best_params_["knn__n_neighbors"]}')
#     # best_model = grid_search.best_estimator_
#     # joblib.dump(best_model, 'model/knn_z-score.sav')
#
#     y_pred = knn_zero_model.predict(x_test_zscore)
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write(f'Akurasi: {accuracy}')
#
#     # results = grid_search.cv_results_
#     # accuracy_scores = results['mean_test_score']
#     # hyperparameters = results['params']
#     # results_df = pd.DataFrame({'Hyperparameters': hyperparameters, 'Accuracy': accuracy_scores})
#     # st.write(results_df)
#
#     # DENGAN PCA =================================================
#
#     knn_zero_PCA_model = joblib.load('model/knn_zero_grid_PCA.pkl')
#     st.write('Dengan PCA')
#     st.success(f'Parameter terbaik K: 6, PCA: 20')
#
#     # st.success(f'Parameter terbaik: \n\n'
#     #            f'K   :{grid_search.best_params_["knn__n_neighbors"]}\n\n'
#     #            f'PCA :{grid_search.best_params_["pca__n_components"]}')
#     # best_model = grid_search.best_estimator_
#     # joblib.dump(best_model, 'model/knn_z-score_PCA-20.sav')
#
#     pca = joblib.load('model/PCA_zero.pkl')
#     pca.transform(x_test_zscore)
#     y_pred = knn_zero_PCA_model.predict(x_test_zscore)
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write(f'Akurasi: {accuracy}')
#
#     # results = grid_search.cv_results_
#     # accuracy_scores = results['mean_test_score']
#     # hyperparameters = results['params']
#     # results_df = pd.DataFrame({'Hyperparameters': hyperparameters, 'Accuracy': accuracy_scores})
#     # st.write(results_df)
#
# def knn_minmax():
#     st.subheader('MinMax')
# #     # Preprocessing Min-Max Scaler
#     x_train_mmax = joblib.load('model/df_train_mmax.sav')
#     scaler = joblib.load('model/mmax_scaler.pkl')
#     x_test_mmax = scaler.transform(x_test)
#
#     knn_mmax_model = joblib.load('model/knn_mmax_grid.pkl')
#     st.write('Tanpa PCA')
#     st.success(f'Parameter terbaik K: 5')
#
# #     st.success(f'Parameter terbaik: \n\nK :{grid_search.best_params_["knn__n_neighbors"]}')
# #     best_model = grid_search.best_estimator_
# #     joblib.dump(best_model, 'model/knn_mmax.sav')
#
#     y_pred = knn_mmax_model.predict(x_test_mmax)
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write(f'Akurasi: {accuracy}')
#
# #     y_pred = best_model.predict(x_test_mmax)
# #     accuracy = accuracy_score(y_test, y_pred)
# #     st.write(f'Akurasi: {accuracy}')
# #     results = grid_search.cv_results_
# #     accuracy_scores = results['mean_test_score']
# #     hyperparameters = results['params']
# #     results_df = pd.DataFrame({'Hyperparameters': hyperparameters, 'Accuracy': accuracy_scores})
# #     st.write(results_df)
# #
# #     # DENGAN PCA =================================================
#
#     knn_zero_PCA_model = joblib.load('model/knn_mmax_grid_PCA.pkl')
#     st.write('Dengan PCA')
#     st.success(f'Parameter terbaik K: 5, PCA: 20')
#
# #     st.write('Dengan PCA')
# #     st.success(f'Parameter terbaik: \n\n'
# #                f'K   :{grid_search.best_params_["knn__n_neighbors"]}\n\n'
# #                f'PCA :{grid_search.best_params_["pca__n_components"]}')
# #     best_model = grid_search.best_estimator_
# #     joblib.dump(best_model, 'model/knn_mmax_PCA-8.sav')
#
#     pca = joblib.load('model/PCA_mmax.pkl')
#     pca.transform(x_test_mmax)
#     y_pred = knn_zero_PCA_model.predict(x_test_mmax)
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write(f'Akurasi: {accuracy}')
#
# #     y_pred = best_model.predict(x_test_mmax)
# #     accuracy = accuracy_score(y_test, y_pred)
# #     st.write(f'Akurasi: {accuracy}')
# #     results = grid_search.cv_results_
# #     accuracy_scores = results['mean_test_score']
# #     hyperparameters = results['params']
# #     results_df = pd.DataFrame({'Hyperparameters': hyperparameters, 'Accuracy': accuracy_scores})
# #     st.write(results_df)
#
