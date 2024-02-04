import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import io


# Load the pre-trained model
# model_path = '/skin_disease_detection_model.h5'

script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = 'skin_disease_detection_model.h5'
model_path = os.path.join(script_dir, model_filename)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")


model = load_model(model_path)

# Define class labels
class_labels = ["Acne", "actinic keratosis", "atypical melanocytic proliferation", "basal cell carcinoma",
                "cafe-au-lait macule","dermatofibroma","lentigo NOS","lichenoid keratosis","melanoma",
                "nevus","other","pigmented benign keratosis","scar","seborrheic keratosis","solar lentigo",
                "squamous cell carcinoma","vascular lesion"]

def predict_skin_disease(image_file):
    try:
         # Convert FileStorage to io.BytesIO
        image_bytes = io.BytesIO(image_file.read())
        
        # Load and preprocess the image
        img = image.load_img(image_bytes, target_size=(128, 128))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make a prediction
        prediction = model.predict(img)
        
        #class_index = np.argmax(prediction)
        #disease_label = class_labels[class_index]
        #accuracy_percentage = prediction[0][class_index] * 100
           
        predictions = []
        for i, label in enumerate(class_labels):
            probability = prediction[0][i] * 100
            predictions.append({"disease": label, "probability": probability})


        return {
            #"predicted_disease": disease_label,
            #"prediction_accuracy": accuracy_percentage
            "predictions": predictions
            
        }

    except Exception as e:
        return {"error": str(e)}
