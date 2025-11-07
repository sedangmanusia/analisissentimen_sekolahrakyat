import os
import random
import streamlit as st
import pymongo
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.utils import resample, class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
import pickle

#inisialisasi seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
# (optional) tf.config.experimental.enable_op_determinism()  # bisa aktifkan kalau perlu

#koneksi ke database
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])

def run():
    client = init_connection()
    db = client["db_sentimen"]

    coll_train = db["data_pelatihan"]
    coll_val = db["data_validasi"]
    coll_test = db["data_pengujian"]
    coll_kamus = db["kamus_normalisasi"]

    st.title("📊 Klasifikasi Sentimen")

    #load database
    def load_collection_as_df(coll):
        docs = list(coll.find())
        if not docs:
            return pd.DataFrame()
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df

    df_train_raw = load_collection_as_df(coll_train)
    df_val_raw = load_collection_as_df(coll_val)
    df_test_raw = load_collection_as_df(coll_test)
    df_kamus = load_collection_as_df(coll_kamus)


    if df_train_raw.empty:
        st.error("Data training tidak ditemukan. Upload data dulu di halaman Upload.")
        st.stop()

    #preprocessing
    kamus = {}
    if not df_kamus.empty:
        if "tidak_baku" in df_kamus.columns and "baku" in df_kamus.columns:
            kamus = dict(zip(df_kamus["tidak_baku"], df_kamus["baku"]))

    stop_words = set(stopwords.words("indonesian"))
    stemmer = StemmerFactory().create_stemmer()

    def remove_noise(text):
        text = str(text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text) #hapus link
        text = re.sub(r"@\w+", "", text) #hapus mention
        text = re.sub(r"#\w+", "", text) #hapus hashtag
        text = re.sub(r"\d+", "", text) #hapus angka
        text = re.sub(r"[^a-zA-Z\s]", " ", text) #hapus semua kecuali huruf
        text = text.translate(str.maketrans("", "", string.punctuation)) #hapus tanda baca
        text = re.sub(r"\s+", " ", text).strip() #memastikan spasi 
        return text

    def tokenize(text):
        return word_tokenize(text)

    def normalize_tokens(tokens): return [kamus.get(t, t) for t in tokens]
    def remove_stop(tokens): return [t for t in tokens if t not in stop_words]
    def do_stem(tokens): return [stemmer.stem(t) for t in tokens]

    def preprocess_pipeline(df):
        df = df.copy().reset_index(drop=True)
        df["lower_text"] = df["full_text"].str.lower()
        df["clean_text"] = df["lower_text"].apply(remove_noise)
        df["tokens"] = df["clean_text"].apply(tokenize)
        df["tokens_norm"] = df["tokens"].apply(normalize_tokens)
        df["tokens_no_stop"] = df["tokens_norm"].apply(remove_stop)
        df["tokens_stemmed"] = df["tokens_no_stop"].apply(do_stem)
        df["final_text"] = df["tokens_stemmed"].apply(lambda x: " ".join(x))
        return df

    st.subheader("Preprocessing Data")

    #Check apakah sudah ada hasil preprocessing
    coll_train_clean = db["data_pelatihan_clean"]
    coll_val_clean   = db["data_validasi_clean"]
    coll_test_clean  = db["data_pengujian_clean"]

    df_train = load_collection_as_df(coll_train_clean)
    df_val   = load_collection_as_df(coll_val_clean)
    df_test  = load_collection_as_df(coll_test_clean)

    if df_train.empty:
        with st.spinner("Memproses data..."):
            df_train = preprocess_pipeline(df_train_raw)
            df_val   = preprocess_pipeline(df_val_raw)
            df_test  = preprocess_pipeline(df_test_raw)

            #Simpan ke DB
            coll_train_clean.delete_many({})
            coll_val_clean.delete_many({})
            coll_test_clean.delete_many({})
            coll_train_clean.insert_many(df_train.to_dict(orient="records"))
            coll_val_clean.insert_many(df_val.to_dict(orient="records"))
            coll_test_clean.insert_many(df_test.to_dict(orient="records"))

        st.success("✅ Preprocessed data tersimpan ke database.")

    st.dataframe(df_train[[
        "full_text", "lower_text", "clean_text", 
        "tokens", "tokens_norm", "tokens_no_stop", 
        "tokens_stemmed", "final_text", "aktual_label"
    ]])

    label_mapping = {"positif": 1, "negatif": 0}
    reverse_mapping = {1: "positif", 0: "negatif"}

    # buat kolom label_encoded berdasarkan mapping 
    df_train["label_encoded"] = df_train["aktual_label"].map(label_mapping)
    df_val["label_encoded"] = df_val["aktual_label"].map(label_mapping)
    df_test["label_encoded"] = df_test["aktual_label"].map(label_mapping)

    st.subheader("Label Encoding")
    st.dataframe(df_train[["final_text", "aktual_label", "label_encoded"]])

    # menampilkan distribusi data
    st.subheader("Distribusi Data")

    label_counts = df_train["label_encoded"].value_counts().to_dict()
    total_data = len(df_train)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Jumlah Kelas Negatif", value=label_counts.get(0, 0), delta=f"{(label_counts.get(0,0)/total_data)*100:.1f}% dari total")
    with col2:
        st.metric(label="Jumlah Kelas Positif", value=label_counts.get(1, 0), delta=f"{(label_counts.get(1,0)/total_data)*100:.1f}% dari total")

    # tampilkan status keseimbangan data dalam card tambahan
    diff = abs(label_counts.get(0, 0) - label_counts.get(1, 0))
    if diff > 0.3 * total_data:
        st.warning("⚠️ Data terlihat *imbalanced*. Pertimbangkan balancing agar model tidak bias.")
    else:
        st.success("✅ Data cukup seimbang antara kelas positif dan negatif.")


    #menampilkan metode balancing data
    st.subheader("Handling Imbalance")
    bal_option = st.selectbox(
        "Pilih metode balancing:",
        ("Downsampling", "Upsampling", "Class Weight", "Tanpa Balancing"),
        index=3
    )

    if bal_option == "Downsampling":
        st.info("Downsampling: menurunkan jumlah kelas mayoritas agar seimbang dengan minoritas.")
        maj_label = df_train["label_encoded"].value_counts().idxmax()
        min_label = df_train["label_encoded"].value_counts().idxmin()
        min_count = df_train["label_encoded"].value_counts().min()
        maj_df = df_train[df_train["label_encoded"] == maj_label]
        min_df = df_train[df_train["label_encoded"] == min_label]
        maj_down = resample(maj_df, replace=False, n_samples=min_count, random_state=SEED)
        df_train_bal = pd.concat([maj_down, min_df]).sample(frac=1, random_state=SEED)
        class_weights = None
    elif bal_option == "Upsampling":
        st.info("Upsampling: menggandakan kelas minoritas agar setara dengan mayoritas.")
        maj_label = df_train["label_encoded"].value_counts().idxmax()
        min_label = df_train["label_encoded"].value_counts().idxmin()
        maj_df = df_train[df_train["label_encoded"] == maj_label]
        min_df = df_train[df_train["label_encoded"] == min_label]
        min_up = resample(min_df, replace=True, n_samples=len(maj_df), random_state=SEED)
        df_train_bal = pd.concat([maj_df, min_up]).sample(frac=1, random_state=SEED)
        class_weights = None
    elif bal_option == "Class Weight":
        st.info("Class Weight: model dilatih dengan bobot lebih besar pada kelas minoritas, dataset tidak diubah.")
        df_train_bal = df_train
        #hitung bobot kelas otomatis dari distribusi data
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(df_train["label_encoded"]),
            y=df_train["label_encoded"]
        )
        class_weights = dict(enumerate(class_weights))
    else:
        st.info("Tanpa Balancing: data dipakai apa adanya.")
        df_train_bal = df_train
        class_weights = None

    # menampilkan distribusi data setelah balancing
    label_counts_bal = df_train_bal["label_encoded"].value_counts().to_dict()
    total_data_bal = len(df_train_bal)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Jumlah Kelas Negatif", value=label_counts_bal.get(0, 0), delta=f"{(label_counts_bal.get(0,0)/total_data_bal)*100:.1f}% dari total")
    with col2:
        st.metric(label="Jumlah Kelas Positif", value=label_counts_bal.get(1, 0), delta=f"{(label_counts_bal.get(1,0)/total_data_bal)*100:.1f}% dari total")

    # tampilkan status keseimbangan data setelah balancing
    diff_bal = abs(label_counts_bal.get(0, 0) - label_counts_bal.get(1, 0))
    if diff_bal > 0.3 * total_data_bal:
        st.warning("⚠️ Data masih terlihat *imbalanced* setelah balancing.")
    else:
        st.success("✅ Data seimbang setelah balancing.")


    #tokenizer dan padding
    with st.sidebar.expander("⚙️ Pengaturan Tokenizer dan Training"):
        vocab_size = st.number_input("Vocab size", 500, 30000, 1000, 500)
        maxlen = st.number_input("Max sequence length", 20, 500, 50, 10)
        EPOCHS = st.number_input("Epochs", 1, 100, 10)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train_bal["final_text"])

    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train_bal["final_text"]), maxlen=maxlen)
    x_val = pad_sequences(tokenizer.texts_to_sequences(df_val["final_text"]), maxlen=maxlen)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test["final_text"]), maxlen=maxlen)

    y_train = df_train_bal["label_encoded"].values
    y_val = df_val["label_encoded"].values
    y_test = df_test["label_encoded"].values

    #model bi lstm
    st.subheader("Arsitektur Model")
    model = Sequential([
        Embedding(vocab_size, 64, input_length=maxlen),
        Bidirectional(LSTM(8, return_sequences=True)),
        Bidirectional(LSTM(8)),
        Dense(8, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    st.text("\n".join(stringlist))

    #training
    class StreamlitLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            st.write(
                f"Epoch {epoch+1}: "
                f"loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, "
                f"val_loss={logs['val_loss']:.4f}, val_acc={logs['val_accuracy']:.4f}"
            )

    if st.button("Mulai Training"):
        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        st.subheader("Proses Training")
        with st.spinner("Melatih model..."):
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=EPOCHS,
                batch_size=32,
                class_weight=class_weights,
                callbacks=[early_stop, StreamlitLogger()],
                verbose=1
            )

        #Tampilkan hasil training (train & val)
        st.subheader("Hasil Training")
        final_train_acc  = history.history["accuracy"][-1]
        final_val_acc    = history.history["val_accuracy"][-1]
        final_train_loss = history.history["loss"][-1]
        final_val_loss   = history.history["val_loss"][-1]

        col1, col2 = st.columns(2)
        col1.metric("Train Accuracy", f"{final_train_acc:.4f}")
        col2.metric("Validation Accuracy", f"{final_val_acc:.4f}")
        col1.metric("Train Loss", f"{final_train_loss:.4f}")
        col2.metric("Validation Loss", f"{final_val_loss:.4f}")

        #Evaluasi di test set
        st.subheader("Evaluasi Test Set")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        col3, col4 = st.columns(2)
        col3.metric("Test Accuracy", f"{test_acc:.4f}")
        col4.metric("Test Loss", f"{test_loss:.4f}")

        #simpan ke session_state
        st.session_state["last_history"]    = history.history
        st.session_state["last_train_acc"]  = final_train_acc
        st.session_state["last_val_acc"]    = final_val_acc
        st.session_state["last_test_acc"]   = test_acc
        st.session_state["last_train_loss"] = final_train_loss
        st.session_state["last_val_loss"]   = final_val_loss
        st.session_state["last_test_loss"]  = test_loss

        #save model + tokenizer + encoder
        model.save("model_sentimen.h5")
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

        # Simpan mapping label
        with open("label_mapping.pkl", "wb") as f:
            pickle.dump({
                "label_mapping": label_mapping,
                "reverse_mapping": reverse_mapping
            }, f)

        #Simpan hasil prediksi
        y_train_pred = (model.predict(x_train) >= 0.5).astype(int).flatten()
        y_val_pred   = (model.predict(x_val) >= 0.5).astype(int).flatten()
        y_test_pred  = (model.predict(x_test) >= 0.5).astype(int).flatten()

        df_train_bal["predict_label"] = [reverse_mapping[int(y)] for y in y_train_pred]
        df_val["predict_label"]       = [reverse_mapping[int(y)] for y in y_val_pred]
        df_test["predict_label"]      = [reverse_mapping[int(y)] for y in y_test_pred]

        coll_train_pred = db["hasil_pred_train"]
        coll_val_pred   = db["hasil_pred_val"]
        coll_test_pred  = db["hasil_pred_test"]

        for coll, df in [
            (coll_train_pred, df_train_bal),
            (coll_val_pred, df_val),
            (coll_test_pred, df_test)
        ]:
            coll.delete_many({})
            coll.insert_many(df.to_dict(orient="records"))

        #Simpan juga hasil training ke DB biar halaman evaluasi konsisten
        coll_metrics = db["training_result"]
        coll_metrics.delete_many({})
        coll_metrics.insert_one({
            "history": history.history,
            "train_acc": float(final_train_acc),
            "val_acc": float(final_val_acc),
            "test_acc": float(test_acc),
            "train_loss": float(final_train_loss),
            "val_loss": float(final_val_loss),
            "test_loss": float(test_loss),
        })

        st.success("✅ Model, tokenizer, label encoder, hasil prediksi & metrics tersimpan.")

    #menampikan hasil training terakhir dengan sessin state
    if "last_history" in st.session_state:
        st.subheader("History Training ")

        col1, col2, col3 = st.columns(3)
        col1.metric("Train Acc", f"{st.session_state['last_train_acc']*100:.2f}%")
        col2.metric("Val Acc", f"{st.session_state['last_val_acc']*100:.2f}%")
        col3.metric("Test Acc", f"{st.session_state['last_test_acc']*100:.2f}%")

        # test akurasi chart
        num_epochs = len(st.session_state["last_history"]["accuracy"])
        test_acc_line = [st.session_state["last_test_acc"]] * num_epochs

        st.line_chart({
            "Train Acc": st.session_state["last_history"]["accuracy"],
            "Val Acc": st.session_state["last_history"]["val_accuracy"],
            "Test Acc": test_acc_line
        })

        # chart test loss
        num_epochs = len(st.session_state["last_history"]["loss"])
        test_loss_line = [st.session_state["last_test_loss"]] * num_epochs

        st.line_chart({
            "Train Loss": st.session_state["last_history"]["loss"],
            "Val Loss": st.session_state["last_history"]["val_loss"],
            "Test Loss": test_loss_line
        })
    else:
        st.info("Belum ada hasil training. Tekan tombol 'Mulai Training' untuk memulai.")