from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import pickle
import base64
import logging
import tf_keras as keras
from tf_keras.models import model_from_json
import pyttsx3
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global classifier
classifier = None
model_loaded = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    global classifier, model_loaded
    model_path = os.path.join(BASE_DIR, 'model', 'model.json')
    weights_path = os.path.join(BASE_DIR, 'model', 'model_weights.h5')
    if os.path.exists(model_path) and os.path.exists(weights_path):
        with open(model_path, "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights(weights_path)
        try:
            classifier._make_predict_function()
        except AttributeError:
            pass  # Not needed in TF2
        model_loaded = True
        logger.info("Model loaded successfully")
    else:
        logger.error("Model files not found at %s", model_path)

def speak_result(msg):
    """Speak the result in a separate thread to avoid blocking"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        if msg == 'Fake':
            engine.say('The currency is Fake')
        else:
            engine.say('The currency is Real')
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("Voice error: " + str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        features_path = os.path.join(BASE_DIR, 'model', 'features.txt.npy')
        labels_path = os.path.join(BASE_DIR, 'model', 'labels.txt.npy')
        X_train = np.load(features_path)
        Y_train = np.load(labels_path)
        return jsonify({
            'status': 'success',
            'message': 'Dataset Loaded. Total images found in dataset for training = ' + str(X_train.shape[0])
        })
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'Dataset files not found. Please run test.py first.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Error loading dataset: ' + str(e)})

@app.route('/generate_model', methods=['POST'])
def generate_model():
    global classifier, model_loaded
    load_model()
    if model_loaded:
        history_path = os.path.join(BASE_DIR, 'model', 'history.pckl')
        with open(history_path, 'rb') as f:
            data = pickle.load(f)
        acc = data['accuracy']
        accuracy = acc[-1] * 100
        return jsonify({
            'status': 'success',
            'message': 'CNN Training Model Accuracy = {:.2f}%'.format(accuracy)
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Model file not found. Please train the model first.'
        })

@app.route('/predict', methods=['POST'])
def predict():
    global classifier, model_loaded
    if not model_loaded:
        return jsonify({'status': 'error', 'message': 'Please generate the model first!'})

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})

    # Read image from upload
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'status': 'error', 'message': 'Invalid image file'})

    # Preprocess for prediction
    img_resized = cv2.resize(img, (64, 64))
    im2arr = np.array(img_resized)
    im2arr = im2arr.reshape(1, 64, 64, 3)
    XX = np.asarray(im2arr)
    XX = XX.astype('float32')
    XX = XX / 255
    preds = classifier.predict(XX)
    predict_val = np.argmax(preds)
    confidence = float(np.max(preds)) * 100

    # Draw result on image
    img_display = cv2.resize(img, (450, 450))
    msg = 'Fake' if predict_val == 0 else 'Real'
    color = (0, 0, 255) if msg == 'Fake' else (0, 255, 0)
    cv2.putText(img_display, '{} ({:.1f}%)'.format(msg, confidence), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Voice feedback in background thread
    voice_thread = threading.Thread(target=speak_result, args=(msg,), daemon=True)
    voice_thread.start()

    # Encode image to base64 for web display
    _, buffer = cv2.imencode('.jpg', img_display)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    logger.info("Prediction: %s (confidence: %.1f%%)", msg, confidence)

    return jsonify({
        'status': 'success',
        'result': msg,
        'confidence': round(confidence, 2),
        'message': 'Currency is detected as: {} (Confidence: {:.1f}%)'.format(msg, confidence),
        'image': img_base64
    })

@app.route('/graph', methods=['POST'])
def graph():
    try:
        history_path = os.path.join(BASE_DIR, 'model', 'history.pckl')
        with open(history_path, 'rb') as f:
            data = pickle.load(f)

        accuracy = data['accuracy']
        loss = data['loss']

        return jsonify({
            'status': 'success',
            'accuracy': [float(a) for a in accuracy],
            'loss': [float(l) for l in loss]
        })
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'Training history not found. Please train the model first.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Error loading graph data: ' + str(e)})

if __name__ == '__main__':
    print("\n  Fake Currency Detection Web App")
    print("  Running at: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
