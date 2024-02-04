from flask import Flask
from flask import Flask, request, jsonify
from model_utils import predict_skin_disease

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Make a prediction using the encapsulated function
        result = predict_skin_disease(image_file)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True, port=5000)