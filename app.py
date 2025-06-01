import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load model & data ===
model = joblib.load('logistic_model.pkl')  # pastikan file ini ada
df_all = pd.read_csv("data.csv")  # pastikan file ini ada
df_all.columns = df_all.columns.str.strip()  # bersihkan nama kolom
df_all['HighSpender'] = (df_all['Spending Score (1-100)'] >= 50).astype(int)  # buat target

# === Judul Aplikasi ===
st.title("Prediksi Pelanggan High Spender")

# === Input Pengguna ===
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Umur", min_value=10, max_value=100, value=25)
income = st.number_input("Pendapatan Tahunan (k$)", min_value=1, max_value=200, value=50)

# Encode Gender
gender_encoded = 1 if gender.lower() == 'male' else 0

# === Tombol Prediksi ===
if st.button("Prediksi"):
    features = np.array([[gender_encoded, age, income]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Output Prediksi
    st.subheader("Hasil Prediksi:")
    st.write("💡", "High Spender" if prediction == 1 else "Low Spender")
    st.write("📊 Probabilitas:", f"{round(probability * 100, 2)}%")

    # === Visualisasi ===
    # Tambahkan input user ke data visualisasi
    user_row = pd.DataFrame({
        'Age': [age],
        'Annual Income (k$)': [income],
        'HighSpender': ['Input']
    })

    plot_df = pd.concat([
        df_all[['Age', 'Annual Income (k$)', 'HighSpender']],
        user_row
    ], ignore_index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df,
        x='Age',
        y='Annual Income (k$)',
        hue='HighSpender',
        palette={'Input': 'black', 0: 'blue', 1: 'red'},
        style='HighSpender',
        s=100
    )
    plt.title('Distribusi Pelanggan & Input User')
    plt.legend(title='Kategori')
    st.pyplot(fig)
