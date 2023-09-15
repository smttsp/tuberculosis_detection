# import sys
#
# from model_cnn.inference import get_model_prediction
# from flask import Flask, request, jsonify
#
#
# if __name__ == "__main__":
#     image_path = sys.argv[1]
#     print(image_path)
#     # image_path = "/Users/samet/Documents/data/TB_Chest_Radiography_Database/Normal/Normal-1821.png"
#     get_model_prediction(image_path)
#


from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model_cnn.inference import get_model_prediction


import string
import random


# initializing size of string
def random_string(k):
    res = "".join(random.choices(string.ascii_lowercase + string.digits, k=k))
    return res


app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    # Assuming the image is sent as a file in the request
    image = request.files.get("image")

    if not image:
        return jsonify({"error": "No image provided"}), 400

    image_path = f"{random_string(8)}.png"
    print(image_path)
    image.save(image_path)  # save the image temporarily

    # Here you'd use your existing get_model_prediction function
    prediction = get_model_prediction(image_path)

    # Clean up
    os.remove(image_path)

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
