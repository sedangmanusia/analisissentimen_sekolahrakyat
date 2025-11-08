import streamlit as st
import numpy as np
import pickle
import json
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def run():
    st.title("Halaman Cek Sentimen")

    # Cek apakah model ada
    if not os.path.exists("model_sentimen.keras"):
        st.error("Model tidak ditemukan. Pastikan model sudah dilatih di halaman Klasifikasi.")
        st.stop()

    try:
        with st.spinner("Memuat model..."):
            model = load_model("model_sentimen.keras")
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}. Pastikan model sudah dilatih dengan benar di halaman Klasifikasi.")
        st.stop()

    try:
        with open("tokenizer.json", "r") as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
    except FileNotFoundError:
        st.error("Tokenizer tidak ditemukan. Pastikan model sudah dilatih di halaman Klasifikasi.")
        st.stop()

    with open("label_mapping.pkl", "rb") as f:
            maps = pickle.load(f)
            #label_mapping = maps["label_mapping"]
            #reverse_mapping = maps["reverse_mapping"]

    # Load maxlen dari config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        maxlen = config.get("maxlen", 50)
    except FileNotFoundError:
        maxlen = 50  # default jika config tidak ada

    user_input = st.text_area("Masukkan Opini Anda Disini")

    if st.button("Cek Sentimen"):
        if user_input.strip() != "":
            text = user_input.lower()

            seq = tokenizer.texts_to_sequences([text])
            pad = pad_sequences(seq, maxlen=maxlen)

            prob = model.predict(pad)[0][0]
            label = "Positif" if prob >= 0.5 else "Negatif"

            st.success(f"Hasil Prediksi:{label}")
            st.write(f"Probabilitas:{prob:.4f}")
        else:
            st.warning("Harap Masukkan Opini Anda")
