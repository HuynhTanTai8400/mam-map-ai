from flask import Flask, request, jsonify, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io
import json

import base64
from flask import render_template

import requests




# #model tren local
# MODEL_PATH = "models/base_model_trained.keras"

# #model trên drive
# MODEL_URL = "https://drive.google.com/uc?export=download&id=1k1B8xe-aYLxu6vVGzhEfN1fjVwqHXbTC"
# def download_model():
#     if not os.path.exists(MODEL_PATH):
#         print("Downloading model...")
#         response = requests.get(MODEL_URL)
#         os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
#         with open(MODEL_PATH, "wb") as f:
#             f.write(response.content)
#         print("Model downloaded.")

# download_model()


# model tren local
MODEL_PATH = "models/base_model_trained.keras"

# model trên drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1k1B8xe-aYLxu6vVGzhEfN1fjVwqHXbTC"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully to {MODEL_PATH}.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            # Optionally, re-raise the exception or exit if model download is critical
            raise RuntimeError(f"Failed to download model from {MODEL_URL}") from e
    else:
        print(f"Model already exists at {MODEL_PATH}.")


# Gọi hàm download_model() trước khi load_model
try:
    download_model()
    # Sau khi tải xong, load model từ đường dẫn cục bộ
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    # Thoát ứng dụng nếu model không thể tải/load
    exit(1) # Rất quan trọng để ứng dụng không chạy mà không có model


# Load model trên serverserver
model = load_model(MODEL_URL)

# Load class names
classes = [
    'Bánh bèo',
    'Bánh bột lọc',
    'Bánh căn',
    'Bánh canh',
    'Bánh chưng',
    'Bánh cuốn',
    'Bánh đúc',
    'Bánh giò',
    'Bánh khọt',
    'Bánh mì',
    'Bánh pía',
    'Bánh tét',
    'Bánh tráng nướng',
    'Bánh xèo',
    'Bún bò Huế',
    'Bún đậu mắm tôm',
    'Bún mắm',
    'Bún riêu',
    'Bún thịt nướng',
    'Cá kho tộ',
    'Canh chua',
    'Cao lầu',
    'Cháo lòng',
    'Cơm tấm',
    'Gỏi cuốn',
    'Hủ tiếu',
    'Mì Quảng',
    'Nem chua',
    'Phở',
    'Xôi xéo'
]

# Khởi tạo Flask app
app = Flask(__name__)

# Load mock metadata từ JSON/local file (hoặc sau này từ DB)
with open("restaurant_data.json", "r", encoding="utf-8") as f:
    restaurants_by_label = json.load(f)


def preprocess_image(file):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((300, 300))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect('/')

    file = request.files['file'].read()
    img_input = preprocess_image(file)
    preds = model.predict(img_input)[0]

    index = int(np.argmax(preds))
    label = classes[index]
    confidence = float(preds[index]) * 100

    # Encode uploaded image to display
    img_base64 = base64.b64encode(file).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{img_base64}"

    # Suggest restaurants
    suggestions = restaurants_by_label.get(label, [])

    return render_template("index.html", result={
        "label": label,
        "confidence": confidence,
        "restaurants": suggestions,
        "image": img_url
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) # Lấy cổng từ biến môi trường PORT, mặc định là 5000
    app.run(host="0.0.0.0", port=port, debug=False) # Rất quan trọng: host="0.0.0.0" và debug=False cho production
