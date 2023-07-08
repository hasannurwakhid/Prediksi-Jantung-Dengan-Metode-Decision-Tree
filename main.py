import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

# Title
st.write('Nama : Hasan Nur Wakhid')
st.write('NIM : A11.2021.13853')
st.write('Kelp : A11.4404')
st.title('Prediksi Terkena Penyakit Jantung')
st.write('Gagal jantung adalah kondisi medis yang terjadi ketika jantung tidak dapat memompa darah dengan efektif seperti yang seharusnya. Ini terjadi ketika otot jantung yang bertanggung jawab untuk memompa darah mengalami kerusakan atau melemah, sehingga tidak mampu memenuhi kebutuhan darah tubuh secara optimal.')
st.write('Silahkan isi form berikut: ')

# Membaca dataset
data = pd.read_csv("heart.csv")

# Memisahkan fitur dan label
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Inisialisasi MinMaxScaler untuk fitur-fitur numerik
scaler = MinMaxScaler()

# Inisialisasi OneHotEncoder untuk variabel kategorikal
encoder = OneHotEncoder(drop='first')

# Menentukan kolom-kolom yang akan dilakukan normalisasi dan encoding
numerical_features = ['Age','RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex','ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Membuat transformer untuk normalisasi dan encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_features),
        ('cat', encoder, categorical_features)])


# Normalisasi dan encoding pada fitur-fitur
X = preprocessor.fit_transform(X)

# Membuat dan melatih model Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Input
Age = st.text_input("Masukkan usia: ")

Sex = st.selectbox("Masukkan jenis kelamin anda:", ('Laki-laki', 'Perempuan'))
if Sex == 'Laki-laki':
    Sex = 'M'
elif Sex == 'Perempuan':
    Sex = 'F'

ChestPainType = st.selectbox("Masukkan jenis nyeri dada", ('Typical Angina', 'Atypical Angina', 'Tidak Memiliki nyeri dada', 'Asymptomatic'))
if ChestPainType == 'Typical Angina':
    ChestPainType = 'TA'
elif ChestPainType == 'Atypical Angina':
    ChestPainType = 'ATA'
elif ChestPainType == 'Tidak Memiliki nyeri dada':
    ChestPainType = 'NAP'
elif ChestPainType == 'Asymptomatic':
    ChestPainType = 'ASY'

RestingBP = st.text_input("Masukkan tekanan darah saat istirahat: ")
Cholesterol = st.text_input("Masukkan kolesterol: ")

FastingBS = st.selectbox("Apakah gula darah lebih dari 120 mg/dl saat berpuasa", ('Ya', 'Tidak'))
if FastingBS == 'Ya':
    FastingBS = '1'
elif FastingBS == 'Tidak':
    FastingBS = '0'

RestingECG = st.selectbox("Masukkan elektrokardiografi saat istirahat", ('Normal', 'Mempunyai ST-T gelombang tidak normal (gelombang T inversi dan atau ST elevasi atau depresi lebih dari 0.05 mV)', 'Menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes'))
if RestingECG == 'Normal':
    RestingECG = 'Normal'
elif RestingECG == 'Mempunyai ST-T gelombang tidak normal (gelombang T inversi dan atau ST elevasi atau depresi lebih dari 0.05 mV)':
    RestingECG = 'ST'
elif RestingECG == 'Menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes':
    RestingECG = 'LVH'

MaxHR = st.text_input("Masukkan denyut jantung maksimal: ")

ExerciseAngina = st.selectbox("Apakah anda mengalami nyeri dada yang disebabkan oleh aktivitas", ('Ya', 'Tidak'))
if ExerciseAngina == 'Ya':
    ExerciseAngina = 'Y'
elif ExerciseAngina == 'Tidak':
    ExerciseAngina = 'N'

Oldpeak = st.text_input("Masukkan ST depression yang diinduksi oleh olahraga relatif terhadap istirahat: ")

ST_Slope = st.selectbox("Masukkan slope segmen ST", ('Upsloping', 'Flat', 'Downsloping'))
if ST_Slope == 'Upsloping':
    ST_Slope = 'Up'
elif ST_Slope == 'Flat':
    ST_Slope = 'Flat'
elif ST_Slope == 'Downsloping':
    ST_Slope = 'Down'

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
    
    # Melakukan normalisasi dan encoding pada input
    input_df = preprocessor.transform(input_df)

    # Perform prediction
    pred_heart = model.predict(input_df)

    if pred_heart == 0:
        st.success('Tidak Memiliki Penyakit Jantung')
    else:
        st.error('Memiliki Penyakit Jantung')