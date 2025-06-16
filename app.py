from flask import Flask, request, jsonify, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io
import json
import base64
import requests

# Đường dẫn lưu model trên local
MODEL_PATH = "models/base_model_trained.keras"

# URL của model trên Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1k1B8xe-aYLxu6vVGzhEfN1fjVwqHXbTC"

def download_model():
    """
    Hàm này kiểm tra và tải model từ Google Drive nếu nó chưa tồn tại.
    """
    if not os.path.exists(MODEL_PATH):
        print("Bắt đầu tải model...")
        # Tạo thư mục 'models' nếu chưa có
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            # Tải model bằng cách sử dụng stream để xử lý file lớn
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Báo lỗi nếu HTTP response là 4xx hoặc 5xx

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model đã được tải thành công về {MODEL_PATH}.")
        except requests.exceptions.RequestException as e:
            print(f"Lỗi khi tải model: {e}")
            # Rất quan trọng: Báo lỗi và dừng ứng dụng nếu không tải được model
            raise RuntimeError(f"Không thể tải model từ {MODEL_URL}. Vui lòng kiểm tra URL hoặc kết nối mạng.") from e
    else:
        print(f"Model đã tồn tại tại {MODEL_PATH}.")


# Gọi hàm tải model
download_model()

# Tải model vào bộ nhớ sau khi đảm bảo nó đã được tải xuống
try:
    model = load_model(MODEL_PATH)
    print("Model đã được load thành công.")
except Exception as e:
    print(f"Lỗi khi load model từ {MODEL_PATH}: {e}")
    # Báo lỗi và dừng ứng dụng nếu không load được model
    raise RuntimeError(f"Không thể load model từ {MODEL_PATH}. Vui lòng kiểm tra file model.") from e

# Load tên các lớp (class names)
classes = [
    'Bánh bèo', 'Bánh bột lọc', 'Bánh căn', 'Bánh canh', 'Bánh chưng',
    'Bánh cuốn', 'Bánh đúc', 'Bánh giò', 'Bánh khọt', 'Bánh mì',
    'Bánh pía', 'Bánh tét', 'Bánh tráng nướng', 'Bánh xèo', 'Bún bò Huế',
    'Bún đậu mắm tôm', 'Bún mắm', 'Bún riêu', 'Bún thịt nướng', 'Cá kho tộ',
    'Canh chua', 'Cao lầu', 'Cháo lòng', 'Cơm tấm', 'Gỏi cuốn',
    'Hủ tiếu', 'Mì Quảng', 'Nem chua', 'Phở', 'Xôi xéo'
]

# Khởi tạo Flask app
app = Flask(__name__)

# Load mock metadata từ JSON/local file (hoặc sau này từ DB)
# Đảm bảo file 'restaurant_data.json' nằm cùng cấp với app.py hoặc trong thư mục gốc của project
try:
    with open("restaurant_data.json", "r", encoding="utf-8") as f:
        restaurants_by_label = json.load(f)
    print("Dữ liệu nhà hàng đã được load thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'restaurant_data.json'. Vui lòng đảm bảo file này tồn tại.")
    restaurants_by_label = {} # Khởi tạo rỗng để tránh lỗi tiếp theo
except json.JSONDecodeError:
    print("Lỗi: Không thể parse file 'restaurant_data.json'. Vui lòng kiểm tra định dạng JSON.")
    restaurants_by_label = {}


def preprocess_image(file):
    """
    Tiền xử lý hình ảnh đầu vào để phù hợp với model.
    """
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((300, 300)) # Thay đổi kích thước về 300x300
    img_array = image.img_to_array(img) / 255.0 # Chuyển đổi thành array và chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch
    return img_array

@app.route("/", methods=["GET"])
def index():
    """
    Hiển thị trang chủ với form tải ảnh lên.
    """
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    #Xử lý yêu cầu dự đoán từ hình ảnh được tải lên.
    if 'file' not in request.files:
        # Nếu không có file trong request, redirect về trang chủ
        return redirect('/')

    file = request.files['file'].read() # Đọc nội dung file ảnh
    img_input = preprocess_image(file) # Tiền xử lý ảnh

    # Dự đoán bằng model
    preds = model.predict(img_input)[0]

    # Lấy nhãn và độ tự tin cao nhất
    index = int(np.argmax(preds))
    label = classes[index]
    confidence = float(preds[index]) * 100

    # Encode ảnh đã tải lên để hiển thị trên trang web
    img_base64 = base64.b64encode(file).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{img_base64}"

    # Đề xuất nhà hàng dựa trên nhãn dự đoán
    suggestions = restaurants_by_label.get(label, [])

    # Trả về kết quả dưới dạng render_template
    return render_template("index.html", result={
        "label": label,
        "confidence": confidence,
        "restaurants": suggestions,
        "image": img_url
    })

if __name__ == "__main__":
    # Lấy cổng từ biến môi trường PORT (được Render.com cung cấp), mặc định là 5000
    port = int(os.environ.get("PORT", 5000))
    # Chạy ứng dụng Flask. Rất quan trọng: host="0.0.0.0" để cho phép kết nối từ bên ngoài
    # và debug=False cho môi trường production.
    app.run(host="0.0.0.0", port=port, debug=False)

