import streamlit as st
import numpy as np
import pickle
import json

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def run():
    st.title("Halaman Cek Sentimen")

    with st.spinner("Memuat model..."):
        model = load_model("model_sentimen.keras")

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

    maxlen = 50

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
