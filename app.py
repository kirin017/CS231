import streamlit as st
from PIL import Image
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from extractfeature import predict_image_mobilenetV2
from joblib import dump, load
from tensorflow.keras.models import load_model
# Tải mô hình đã huấn luyện (thay file model.h5 bằng mô hình của bạn)

LABEL = {0:'Smoke',1:'fire',2:'non fire'}

def load_model_(model_name):
    if model_name == "SVM":
        return load('svm.joblib')  # Đường dẫn model
    elif model_name == "MobileNetV2":
        return load_model('mobilenet_model_best.keras')  
    else:
        raise ValueError("Model không hợp lệ")

def preprocess_image(image, target_size=(224, 224)):
    """
    Tiền xử lý ảnh để đưa vào mô hình.
    - Resize ảnh về kích thước phù hợp.
    - Chuẩn hóa (normalization) nếu cần.
    """
    image_resized = cv2.resize(image, target_size)  # Resize ảnh # Chuyển sang array
    image_resized = np.expand_dims(image_resized, axis=0)  # Thêm chiều batch
    image_resized = image_resized / 255.0  # Chuẩn hóa giá trị pixel
    return image_resized
def extract_feature(img):
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten()
st.title("Ứng dụng Dự đoán Ảnh")
st.write("Tải ảnh lên và nhận nhãn dự đoán từ mô hình.")

model_name = st.sidebar.selectbox("Chọn Model", ["SVM", "MobileNetV2"])   
model = load_model_(model_name)
# Tải ảnh lên
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        # Hiển thị ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Đọc ảnh với OpenCV
    st.image(image,channels = "BGR", caption="Ảnh đã tải lên", use_container_width=True)
    feature = extract_feature(image) 
        # Dự đoán
if st.button("Predict:"):        
    with st.spinner("Đang dự đoán..."):
        #label = load_model(image)
        #label = predict_image_mobilenetV2(image)
        # if model_name == "SVM":
        #     label = model.predict([feature])
        #     print(label)
        # #label = model.predict(feature.reshape(1, -1))
        # else:
        image = preprocess_image(image)
        label = model.predict(image)
        label_id = (np.argmax(label, axis=1))
    st.success(f"Dự đoán:{LABEL[label_id[0]]}")
