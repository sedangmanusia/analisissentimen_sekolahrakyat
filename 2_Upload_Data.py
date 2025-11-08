import streamlit as st
import pymongo
import pandas as pd

# Koneksi ke database
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])

def run():
    st.title("⬆️ Upload Dataset Anda di Sini")

    # Koneksi MongoDB
    client = init_connection()
    db = client["db_sentimen"]

    # Pilihan jenis data
    pilih = st.selectbox(
        "Pilih Jenis Data yang Akan Anda Upload",
        ["Data Pelatihan", "Data Validasi", "Data Pengujian", "Kamus Normalisasi"]
    )

    # Fungsi upload dan simpan
    def upload_save(collection_name, file_label):
        collection = db[collection_name]

        st.warning("⚠️ Upload data baru akan *menghapus semua data lama* pada kategori ini setelah Anda konfirmasi.")

        uploaded_file = st.file_uploader(f"Upload {file_label}", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview Data")
            st.dataframe(df)

            # Tombol simpan data
            if st.button(f"Simpan {file_label} ke Database"):
                # Buka dialog konfirmasi
                confirm_save(collection, df, file_label)

        # Tampilkan data yang sudah ada
        existing_data = list(collection.find())
        if existing_data:
            df_existing = pd.DataFrame(existing_data)
            if "_id" in df_existing.columns:
                df_existing = df_existing.drop(columns="_id")
            st.write(f"{file_label} yang sudah ada dalam database:")
            st.dataframe(df_existing)
        else:
            st.info(f"Belum ada {file_label} yang tersimpan di database.")

    @st.dialog("Konfirmasi Penyimpanan")
    def confirm_save(collection, df, file_label):
        st.write(f"Anda akan menyimpan ulang {file_label}. Semua data lama pada kategori ini akan dihapus permanen.")
        st.write("Apakah Anda yakin ingin melanjutkan?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ya", use_container_width=True):
                # Hapus semua data lama
                collection.delete_many({})

                # Hapus duplikat dalam file yang sama
                df.drop_duplicates(inplace=True)

                # Simpan data baru
                data = df.to_dict("records")
                collection.insert_many(data)

                st.toast(f"{file_label} berhasil disimpan ke database (data lama telah diganti)", icon="✅")
                st.rerun()  # Tutup dialog
        with col2:
            if st.button("Tidak", use_container_width=True):
                st.toast("Penyimpanan dibatalkan.", icon="ℹ️")
                st.rerun()  # Tutup dialog

    # Pemetaan jenis data ke koleksi
    if pilih == "Data Pelatihan":
        upload_save("data_pelatihan", "Data Pelatihan")
    elif pilih == "Data Validasi":
        upload_save("data_validasi", "Data Validasi")
    elif pilih == "Data Pengujian":
        upload_save("data_pengujian", "Data Pengujian")
    elif pilih == "Kamus Normalisasi":
        upload_save("kamus_normalisasi", "Kamus Normalisasi")