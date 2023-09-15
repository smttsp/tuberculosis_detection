import os
import random
import string

from flask import Flask, jsonify, request
from flask_cors import CORS

from model_cnn.inference import get_model_prediction

app = Flask(__name__)
CORS(app)


def get_random_string(k):
    res = "".join(random.choices(string.ascii_lowercase + string.digits, k=k))
    return res


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files.get("image")

    if not image:
        return jsonify({"error": "No image provided"}), 400

    image_path = f"{get_random_string(k=8)}.png"
    print(image_path)
    image.save(image_path)

    prediction = get_model_prediction(image_path)

    os.remove(image_path)

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
