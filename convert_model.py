from tensorflow import keras
import sys

try:
    model = keras.models.load_model("model_sentimen.keras")  # atau .h5 bila masih .h5
    # jika model ada argumen deprecated, anda bisa rebuild sederhana (opsional)
    model.save("model_sentimen_v2.keras", save_format="keras")
    print("✅ Saved model_sentimen_v2.keras")
except Exception as e:
    print("❌ Error:", e)
    sys.exit(1)
