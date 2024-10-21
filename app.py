from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)
loaded_model = tf.keras.models.load_model('saved_models\2.keras')

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.route("/predict",methods=["POST"])
def predict():

    class_names = ["EARLY BLIGHT","LATE BLIGHT","HEALTHY"]
    if 'file' not in request.files:
        return jsonify({"error":"No file found"}),400
    
    file = request.files['file']

    if file.filename=='':
        return jsonify({"error" : "No selected file"}), 400
    
    image = read_file_as_image(file.read())
    image = np.expand_dims(image, axis=0)    
    prediction = loaded_model.predict(image)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    app.run(debug=True)
