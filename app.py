from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('model_new2.h5')  # Ganti dengan nama dan lokasi model Anda

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model Anda
    img_array = np.asarray(img)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    prediction = model.predict(img_array)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
