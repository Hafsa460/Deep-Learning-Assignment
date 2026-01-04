import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
from PIL import Image
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
SEQUENCE_LENGTH = 5
IMAGE_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

ACTIONS = [
    'applauding', 'holding_an_umbrella', 'shooting_an_arrow',
    'blowing_bubbles', 'jumping', 'smoking',
    'brushing_teeth', 'looking_through_a_microscope', 'taking_photos',
    'cleaning_the_floor', 'looking_through_a_telescope', 'texting_message',
    'climbing', 'phoning', 'throwing_frisby',
    'cooking', 'playing_guitar', 'using_a_computer',
    'cutting_trees', 'playing_violin', 'walking_the_dog',
    'cutting_vegetables', 'pouring_liquid', 'washing_dishes',
    'drinking', 'pushing_a_cart', 'watching_TV',
    'feeding_a_horse', 'reading', 'waving_hands',
    'fishing', 'riding_a_bike', 'writing_on_a_board',
    'fixing_a_bike', 'riding_a_horse', 'writing_on_a_book',
    'fixing_a_car', 'rowing_a_boat',
    'gardening', 'running'
]

NUM_CLASSES = len(ACTIONS)

model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_trained_model(model_path):
    global model
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False

        try:
            model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            msg = str(e)
            logger.warning(f"Standard load_model failed: {msg}")
            if "Unrecognized keyword arguments: ['batch_shape']" in msg or "batch_shape" in msg:
                try:
                    from tensorflow.keras.layers import InputLayer as KInputLayer

                    class LegacyInputLayer(KInputLayer):
                        @classmethod
                        def from_config(cls, config):
                            # remove unsupported keys added by older Keras versions
                            config = dict(config)
                            config.pop('batch_shape', None)
                            return super(LegacyInputLayer, cls).from_config(config)

                    model = load_model(model_path, custom_objects={'InputLayer': LegacyInputLayer})
                    logger.info(f"Model loaded successfully (legacy InputLayer fallback) from {model_path}")
                    return True
                except Exception as e2:
                    logger.error(f"Fallback load failed: {str(e2)}")
                    return False
            else:
                return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


def ensure_model_loaded():
    global model
    if model is not None:
        return True

    model_path = os.getenv('MODEL_PATH') or 'cnn_Lstm_model.h5'
    logger.info(f"Model not loaded — attempting to load from: {model_path}")
    ok = load_trained_model(model_path)
    if not ok:
        for fname in os.listdir('.'):
            if fname.lower().endswith('.h5'):
                logger.info(f"Attempting fallback load of discovered model file: {fname}")
                if load_trained_model(fname):
                    return True
        logger.warning("ensure_model_loaded failed: no compatible model found or load error")
        return False

    return True

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path) 
        if img is None:
            return None
        img = cv2.resize(img, IMAGE_SIZE)  
        img = img.astype("float32") / 255.0
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def preprocess_image_from_bytes(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, IMAGE_SIZE)  
        img = img.astype("float32") / 255.0
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image from bytes: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_actions': NUM_CLASSES
    }), 200

@app.route('/actions', methods=['GET'])
def get_actions():
    return jsonify({
        'actions': ACTIONS,
        'num_actions': NUM_CLASSES
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        if not ensure_model_loaded():
            return jsonify({'error': 'Model not loaded', 'hint': 'Upload model via /load-model or set MODEL_PATH and restart, or place cnn_Lstm_model.h5 in backend folder'}), 503

    if 'files' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files') if 'files' in request.files else [request.files['file']]
    
    if len(files) == 0:
        return jsonify({'error': 'No files provided'}), 400

    if len(files) < SEQUENCE_LENGTH:
        return jsonify({
            'error': f'Please provide at least {SEQUENCE_LENGTH} images. Provided: {len(files)}'
        }), 400

    try:
        frames = []
        for file in files[:SEQUENCE_LENGTH]: 
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'Invalid file type: {file.filename}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400

            image_bytes = file.read()
            img = preprocess_image_from_bytes(image_bytes)
            
            if img is None:
                return jsonify({'error': f'Failed to process image: {file.filename}'}), 400
            
            frames.append(img)

        if len(frames) < SEQUENCE_LENGTH:
            return jsonify({
                'error': f'Could not process all images. Successfully processed: {len(frames)}/{SEQUENCE_LENGTH}'
            }), 400

        X = np.array([frames])  

        predictions = model.predict(X, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'action': ACTIONS[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]

        return jsonify({
            'predicted_action': ACTIONS[pred_class],
            'confidence': confidence,
            'all_predictions': top_3_predictions,
            'num_images_processed': len(frames)
        }), 200

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in prediction: {str(e)}\n{tb}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/predict-single', methods=['POST'])
def predict_single():
    if model is None:
        if not ensure_model_loaded():
            return jsonify({'error': 'Model not loaded', 'hint': 'Upload model via /load-model or set MODEL_PATH and restart, or place cnn_Lstm_model.h5 in backend folder'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        image_bytes = file.read()
        img = preprocess_image_from_bytes(image_bytes)

        if img is None:
            return jsonify({'error': 'Failed to process image'}), 400

        frames = [img] * SEQUENCE_LENGTH
        X = np.array([frames])

        predictions = model.predict(X, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'action': ACTIONS[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]

        return jsonify({
            'predicted_action': ACTIONS[pred_class],
            'confidence': confidence,
            'all_predictions': top_3_predictions,
            'note': 'Single image was repeated to create sequence'
        }), 200

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in single prediction: {str(e)}\n{tb}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    global model

    if 'model' in request.files:
        f = request.files['model']
        if f.filename == '':
            return jsonify({'error': 'No model file selected'}), 400
        fname = f.filename
        if not fname:
            return jsonify({'error': 'No model file selected'}), 400
        filename = secure_filename(str(fname))
        save_path = os.path.join('.', filename)
        f.save(save_path)
        ok = load_trained_model(save_path)
        if ok:
            return jsonify({'status': 'model_loaded', 'path': save_path}), 200
        else:
            return jsonify({'error': 'Failed to load uploaded model'}), 500

    model_path = request.form.get('model_path') or os.getenv('MODEL_PATH') or 'cnn_Lstm_model.h5'
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model path does not exist', 'model_path': model_path}), 400

    ok = load_trained_model(model_path)
    if ok:
        return jsonify({'status': 'model_loaded', 'path': model_path}), 200
    else:
        return jsonify({'error': 'Failed to load model', 'path': model_path}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size: 5 MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    model_path = os.getenv('MODEL_PATH', 'C:\\Users\\syeda\\Documents\\Assignment\\action-recognition-app\\backend\\cnn_Lstm_model.h5')
    
    if not load_trained_model(model_path):
        logger.warning(f"Could not load model from {model_path}")
        found = False
        for fname in os.listdir('.'):
            if fname.lower().endswith('.h5'):
                logger.info(f"Found model file '{fname}' in backend directory — attempting to load it")
                if load_trained_model(fname):
                    found = True
                    break

        if not found:
            logger.warning("No .h5 model found in the backend directory.\n"
                           "Place your 'cnn_Lstm_model.h5' into the backend folder or set the MODEL_PATH environment variable.\n"
                           "Example (PowerShell): $env:MODEL_PATH='C:\\full\\path\\to\\cnn_Lstm_model.h5' ; python app.py")
            logger.warning("API will still run, but predictions will fail until a model is loaded")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
