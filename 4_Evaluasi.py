import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

#Koneksi ke MongoDB
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(
        host=st.secrets["mongo"]["host"],
        port=st.secrets["mongo"]["port"],
        username=st.secrets["mongo"].get("username"),
        password=st.secrets["mongo"].get("password"),
    )

def run():
    client = init_connection()
    db = client["db_sentimen"]

    # load koleksi MongoDB
    def load_collection(coll):
        docs = list(coll.find())
        if not docs:
            return pd.DataFrame()
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df

    st.title("Evaluasi Model")

    #  Ambil hasil prediksi test
    df_result = load_collection(db["hasil_pred_test"])

    if df_result.empty:
        st.error("❌ Data hasil prediksi tidak ditemukan. Harap lakukan training model terlebih dahulu.")
        st.stop()

    #  Load file label mapping 
    try:
        with open("label_mapping.pkl", "rb") as f:
            maps = pickle.load(f)
            label_mapping = maps["label_mapping"]
            reverse_mapping = maps["reverse_mapping"]
    except FileNotFoundError:
        st.error("❌ File 'label_mapping.pkl' tidak ditemukan. Pastikan sudah melakukan training model terlebih dahulu.")
        st.stop()

    st.subheader("Hasil Prediksi")
    # Fungsi untuk styling baris yang salah prediksi
    def highlight_incorrect(row):
        if row['aktual_label'] != row['predict_label']:
            return ['background-color: #ffcccc'] * len(row)
        else:
            return [''] * len(row)

    # Terapkan styling ke dataframe
    styled_df = df_result[["full_text", "aktual_label", "predict_label"]].style.apply(highlight_incorrect, axis=1)
    st.dataframe(styled_df)

    #Konversi label teks ke angka
    y_true = df_result["aktual_label"].map(label_mapping)
    y_pred = df_result["predict_label"].map(label_mapping)

    # Urutkan label berdasarkan nilai numeriknya agar tidak tertukar
    labels_ordered = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]

    #Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_ordered,
        yticklabels=labels_ordered, ax=ax
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    #Perhitungan metrik performa
    st.subheader("Metrik Performa")

    total_data = np.sum(cm)
    benar = np.trace(cm)
    akurasi = benar / total_data

    presisi_kelas = []
    recall_kelas = []
    f1_kelas = []

    for i in range(len(cm)):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_i = (2 * precision_i * recall_i / (precision_i + recall_i)) if (precision_i + recall_i) > 0 else 0

        presisi_kelas.append(precision_i)
        recall_kelas.append(recall_i)
        f1_kelas.append(f1_i)

    #Rata-rata tertimbang
    support = np.sum(cm, axis=1)
    total_support = np.sum(support)
    precision_weighted = np.sum(np.array(presisi_kelas) * support) / total_support
    recall_weighted = np.sum(np.array(recall_kelas) * support) / total_support
    f1_weighted = np.sum(np.array(f1_kelas) * support) / total_support

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{akurasi:.2f}")
    col2.metric("Precision", f"{precision_weighted:.2f}")
    col3.metric("Recall", f"{recall_weighted:.2f}")
    col4.metric("F1 Score", f"{f1_weighted:.2f}")

    # Tampilkan tabel metrik per kelas
    df_metrics = pd.DataFrame({
        "Label": labels_ordered,
        "Precision": presisi_kelas,
        "Recall": recall_kelas,
        "F1-Score": f1_kelas,
        "Support": support
    })

    st.dataframe(df_metrics.style.format({
        "Precision": "{:.2f}",
        "Recall": "{:.2f}",
        "F1-Score": "{:.2f}"
    }))

    # Simpan hasil evaluasi ke koleksi evaluation_result
    eval_data = {
        "akurasi": akurasi,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "metrics_per_class": df_metrics.to_dict(orient="records"),
    }

    db["evaluation_result"].insert_one(eval_data)


    st.success("✅ Evaluasi selesai dan hasil telah disimpan ke database.")
