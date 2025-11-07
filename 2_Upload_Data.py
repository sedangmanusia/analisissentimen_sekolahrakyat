import streamlit as st
import pymongo
import pandas as pd
#import numpy as np

#koneksi ke database
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(
        host=st.secrets["mongo"]["host"],
        port=st.secrets["mongo"]["port"],
        username=st.secrets["mongo"]["username"],
        password=st.secrets["mongo"]["password"],
    )

def run():
    st.title("⬆️Upload Datase Anda Disini")
    client = init_connection()
    db = client["db_sentimen"]

    #selectbox upload data
    pilih = st.selectbox(
        "Pilih Jenis Data Yang Akan Anda Upload",
        ["Data Pelatihan", "Data Validasi", "Data Pengujian", "Kamus Normalisasi"]
    )

    #fungsi upload data
    def upload_save(collection_name, file_label):
        collection = db[collection_name]

        uploaded_file = st.file_uploader(f"Upload {file_label}", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview Data")
            st.dataframe(df)

            #simpan uploaded data ke database
            if st.button(f"Simpan {file_label} ke Database"):
                data = df.to_dict("records")
                result = collection.insert_many(data)
                st.success(f"{file_label} berhasil disimpan ke database")

        #tampilkan data yang sudah ada dalam database
        existing_data = list(collection.find())
        if existing_data:
            df_existing = pd.DataFrame(existing_data)
            if"_id" in df_existing.columns:
                df_existing = df_existing.drop(columns="_id")
            st.write(f"{file_label} yang sudah ada dalam database")
            st.dataframe(df_existing)
        else:
            st.info(f"Belum ada {file_label} yang tersimpan")

    #pilihan 
    if pilih == "Data Pelatihan":
        upload_save("data_pelatihan", "Data Pelatihan")
    elif pilih == "Data Validasi":
        upload_save("data_validasi", "Data Validasi")
    elif pilih == "Data Pengujian":
        upload_save("data_pengujian", "Data Pengujian")
    elif pilih == "Kamus Normalisasi":
        upload_save("kamus_normalisasi", "Kamus Normalisasi")