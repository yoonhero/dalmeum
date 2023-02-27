import base64
import os
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from flask_cors import CORS, cross_origin
from inference import inference


app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config.from_pyfile('config.py')
# api = Api(app)

ALLOWED_EXTENSIONS = {'png', "jpg", "jpeg"}


def allowed_file(filename):
    return '.' in filename and filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
@cross_origin(supports_credentials=True)
def predict():
    print("...")
    params = request.get_json()

    image = params["image"]

    try:
        image = image[image.find(",") + 1:]
        dec = base64.b64decode(image + "===")
        image = Image.open(BytesIO(dec)).convert("RGB")

    except:
        return jsonify({"error": "Error when reading Image."})

    image_url, confidence = inference(image)

    # result = {"player_name": name, "url": image_url}
    result = {"url": image_url, "confidence": confidence}

    return jsonify({
        "statusCode": 200,
        "body": result,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credential": True
        },
    })

if __name__ == "__main__":
    app.run("localhost", 5000, debug=False)