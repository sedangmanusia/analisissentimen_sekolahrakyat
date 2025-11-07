import streamlit as st
import pandas as pd
import altair as alt
import pymongo

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])

def run():
    st.title("📊 Dashboard Analisis Sentimen")

    # proses koneksi & load data
    with st.spinner("🔄 Memuat data dari database..."):
        client = init_connection()
        db = client["db_sentimen"]

        coll_train = db["data_pelatihan"]
        coll_val = db["data_validasi"]
        coll_test = db["data_pengujian"]

        def load_collection(coll):
            docs = list(coll.find())
            if not docs:
                return pd.DataFrame(docs)
            df = pd.DataFrame(docs)
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            return df

        df_train = load_collection(coll_train)
        df_val = load_collection(coll_val)
        df_test = load_collection(coll_test)

        eval_data = db["evaluation_result"].find_one(sort=[("_id", -1)])

    #Hitung distribusi label
    all_labels = pd.concat([
        df_train["aktual_label"],
        df_val["aktual_label"],
        df_test["aktual_label"]
    ])

    total = all_labels.value_counts().to_dict()
    total_positif = total.get(1, 0) or total.get("positif", 0)
    total_negatif = total.get(0, 0) or total.get("negatif", 0)

    #Info hasil evaluasi
    if eval_data:
        st.info(f"Akurasi Model Saat Ini: {eval_data['akurasi']*100:.2f}%")
    else:
        st.warning("Belum ada hasil evaluasi tersimpan. Silakan jalankan halaman Evaluasi terlebih dahulu.")

    #Cards jumlah data
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.metric(label="Jumlah Data Positif 😀", value=total_positif)
    with col2:
        st.metric(label="Jumlah Data Negatif 😠", value=total_negatif)

    #proses render chart
    with st.spinner("📈 Membuat visualisasi chart..."):
        data = pd.DataFrame({
            "Kategori": ["Positif", "Negatif"],
            "Jumlah": [total_positif, total_negatif]
        })

        chart = alt.Chart(data).mark_arc(innerRadius=50).encode(
            theta="Jumlah",
            color=alt.Color("Kategori", scale=alt.Scale(scheme="set2")),
            tooltip=["Kategori", "Jumlah"]
        ).properties(
            width=400,
            height=300,
            title="Distribusi Data Sentimen"
        )

        st.altair_chart(chart, use_container_width=True)