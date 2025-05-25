from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
import gdown

app = Flask(__name__)

# ====== Cấu hình tải model PhoBERT ======
MODEL_DIR = "phobert_model"
FILES = {
    "config.json": "1rInpVFfRnL8VGz4R9GuT2YBDUNm_2Hwo",
    "pytorch_model.bin": "1q2eW_hKoMQ89PYYl8NQ5rQknrZoZvKsM",
    "vocab.txt": "1VNg4eDBon934pkAUQTfDKG-vqXsojiM-",
    "merges.txt": "1ONzvnxWisPlbIFbaQdSsqopJBr6RMN8L",
    "special_tokens_map.json": "137Fe5OixmRaPjIq3-cjFHT_vSSKAmDVI",
    "tokenizer_config.json": "1Gq52RMW8CoY_ZHx3y_wLBHov33ggCK7Y"
}

# Tải các file nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)
for filename, file_id in FILES.items():
    file_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Đang tải {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)

# ====== Load PhoBERT ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# ====== Flask App ======
results = []

def predict_phobert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    global results
    if request.method == "POST":
        action = request.form.get("action")
        if action == "predict":
            input_data = {
                "nghe_cha": request.form.get("nghe_cha"),
                "nghe_me": request.form.get("nghe_me"),
                "gio_hoc": request.form.get("gio_hoc"),
                "gio_choi": request.form.get("gio_choi"),
                "gio_mang": request.form.get("gio_mang"),
                "gpa": request.form.get("gpa"),
                "thich_nghi": request.form.get("thich_nghi"),
                "phuong_phap": request.form.get("phuong_phap"),
                "ho_tro": request.form.get("ho_tro"),
                "co_so": request.form.get("co_so"),
                "chat_luong": request.form.get("chat_luong"),
                "chuong_trinh": request.form.get("chuong_trinh"),
                "tinh_canh_tranh": request.form.get("tinh_canh_tranh"),
                "anh_huong": request.form.get("anh_huong")
            }
            input_text = " ".join(input_data.values())
            prediction = predict_phobert(input_text)
            input_data["ket_qua"] = f"Kết quả: {prediction}"
            results.append(input_data)
        elif action == "save":
            df = pd.DataFrame(results)
            df.to_csv("ket_qua_du_doan.csv", index=False)
        elif action.startswith("delete_"):
            index = int(action.split("_")[1])
            if 0 <= index < len(results):
                results.pop(index)
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
