import streamlit as st
import pandas as pd
import numpy as np

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title('Prediksi Terkena Penyakit Jantung')
st.write('Gagal jantung adalah kondisi medis yang terjadi ketika jantung tidak dapat memompa darah dengan efektif seperti yang seharusnya. Ini terjadi ketika otot jantung yang bertanggung jawab untuk memompa darah mengalami kerusakan atau melemah, sehingga tidak mampu memenuhi kebutuhan darah tubuh secara optimal.')
st.write('Silahkan isi form berikut: ')

col1, col2 = st.columns(2)

# Input
Age = st.text_input("Masukkan usia: ")
Sex = st.selectbox("Masukkan jenis kelamin", ('M', 'F'))
ChestPainType = st.selectbox("Masukkan jenis nyeri dada", ('TA', 'ATA', 'NAP', 'ASY'))
RestingBP = st.text_input("Masukkan tekanan darah saat istirahat: ")
Cholesterol = st.text_input("Masukkan kolesterol: ")
FastingBS = st.selectbox("Masukkan gula darah saat berpuasa", ('1', '0'))
RestingECG = st.selectbox("Masukkan elektrokardiografi istirahat", ('Normal', 'ST', 'LVH'))
MaxHR = st.text_input("Masukkan denyut jantung maksimal: ")
ExerciseAngina = st.selectbox("Masukkan angina yang disebabkan oleh aktivitas", ('Y', 'N'))
Oldpeak = st.text_input("Masukkan ST depression yang diinduksi oleh olahraga relatif terhadap istirahat: ")
ST_Slope = st.selectbox("Masukkan slope segmen ST", ('Up', 'Flat', 'Down'))

# Prediksi
pred_heart = ''

# Tombol untuk prediksi
if st.button('Prediksi'):
    # Konversi nilai input ke tipe numerik
    Age = int(Age)
    RestingBP = float(RestingBP)
    Cholesterol = float(Cholesterol)
    MaxHR = float(MaxHR)
    Oldpeak = float(Oldpeak)

    input_attributes = {
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }

    # Mengubah input menjadi bentuk DataFrame
    input_df = pd.DataFrame([input_attributes])

    # Menentukan kolom-kolom yang akan dilakukan normalisasi dan encoding
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # Encoding menggunakan OneHotEncoder dan Normalisasi menggunakan MinMaxScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)])

    st.write(input_df)

    input_data = preprocessor.fit_transform(input_df)

    st.write(input_data)

    # Perform prediction
    pred_heart = model.predict(input_data)

    if pred_heart == 0:
        st.success('Tidak Memiliki Penyakit Jantung')
    else:
        st.error('Memiliki Penyakit Jantung')
