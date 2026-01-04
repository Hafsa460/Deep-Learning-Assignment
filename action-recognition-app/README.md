# Action Recognition System

A CNN-LSTM based action recognition system with a modern web interface. This application uses deep learning to recognize human actions from image sequences.

## 🎯 Features

- **CNN-LSTM Architecture**: Combines MobileNetV2 for feature extraction with LSTM for temporal analysis
- **41 Action Classes**: Recognizes 41 different human actions
- **Dual Upload Modes**: 
  - Single image mode (image is repeated to create sequence)
  - Multiple image sequence mode (for more accurate predictions)
- **Real-time Predictions**: Get instant action recognition results
- **Top-3 Predictions**: View confidence scores for top 3 predictions
- **Responsive UI**: Works on desktop and mobile devices
- **REST API**: Fully functional API for integration with other applications

## 📋 Recognized Actions

The system can recognize 41 different actions:
- **Basic Actions**: Applauding, Climbing, Drinking, Eating, Feeding a horse, Fishing, Fixing a bike, Fixing a car, Gardening, Holding an umbrella, Jumping, Phoning, Playing guitar, Playing violin, Pouring liquid, Reading, Riding a bike, Riding a horse, Running, Shooting an arrow, Smoking, Taking photos, Texting message, Throwing frisbee, Using a computer, Walking the dog, Watching TV, Waving hands, Writing on a board, Writing on a book
- **Object-based Actions**: Blowing bubbles, Brushing teeth, Cleaning the floor, Cooking, Cutting trees, Cutting vegetables, Looking through a microscope, Looking through a telescope, Pushing a cart, Rowing a boat, Washing dishes

## 🏗️ System Architecture

### Backend (Flask)
- **Port**: 5000
- **Framework**: Flask with TensorFlow/Keras
- **Model**: CNN-LSTM (cnn_Lstm_model.h5)
- **API Endpoints**:
  - `GET /health` - Health check
  - `GET /actions` - List all recognized actions
  - `POST /predict` - Predict from multiple images
  - `POST /predict-single` - Predict from single image

### Frontend (Flask + HTML/CSS/JavaScript)
- **Port**: 3000
- **Features**: Modern UI, drag-and-drop uploads, real-time predictions
- **Communication**: REST API calls to backend

## 📦 Project Structure

```
action-recognition-app/
├── backend/
│   ├── app.py                 # Flask backend API
│   ├── requirements.txt        # Python dependencies
│   └── cnn_Lstm_model.h5      # Trained model (place here)
├── frontend/
│   ├── app.py                 # Flask frontend server
│   ├── templates/
│   │   └── index.html         # Main HTML
│   └── static/
│       ├── style.css          # Styling
│       └── script.js          # JavaScript functionality
├── README.md                   # This file
└── setup.md                    # Detailed setup guide
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- GPU support (optional, but recommended for faster inference)

### Installation

#### 1. Clone or Extract the Project
```bash
cd action-recognition-app
```

#### 2. Set Up Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Important**: Place your trained `cnn_Lstm_model.h5` file in the `backend/` directory.

#### 3. Set Up Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install Flask
pip install Flask

# Or use requirements if you want to keep it consistent:
pip install Flask==2.3.2
```

### 4. Running the Application

#### Start Backend (Terminal 1):
```bash
cd backend
python app.py
```
You should see:
```
 * Running on http://0.0.0.0:5000
```

#### Start Frontend (Terminal 2):
```bash
cd frontend
python app.py
```
You should see:
```
 * Running on http://0.0.0.0:3000
```

#### 5. Access the Application

Open your web browser and navigate to:
```
http://localhost:3000
```

## 🎮 How to Use

### Single Image Mode
1. Go to the "Single Image" tab
2. Upload an image by drag-and-drop or click to select
3. Click "Predict Action"
4. View results with confidence scores

### Multiple Images Mode
1. Go to the "Multiple Images (Sequence)" tab
2. Upload at least 5 images (drag-and-drop multiple files or select them individually)
3. Click "Predict Action"
4. View results with top-3 predictions

### Information Tab
View system status, all recognized actions, and how the system works.

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Get All Actions
```bash
curl http://localhost:5000/actions
```

### Predict from Multiple Images
```bash
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" -F "files=@image3.jpg" \
     -F "files=@image4.jpg" -F "files=@image5.jpg" \
     http://localhost:5000/predict
```

### Predict from Single Image
```bash
curl -X POST -F "file=@image.jpg" \
     http://localhost:5000/predict-single
```

### Response Format
```json
{
  "predicted_action": "reading",
  "confidence": 0.95,
  "all_predictions": [
    {"action": "reading", "confidence": 0.95},
    {"action": "writing_on_a_book", "confidence": 0.03},
    {"action": "watching_TV", "confidence": 0.01}
  ],
  "num_images_processed": 5
}
```

## 🛠️ Configuration

### Backend Configuration (app.py)
- `UPLOAD_FOLDER`: Directory for temporary image storage
- `ALLOWED_EXTENSIONS`: Supported image formats
- `MAX_FILE_SIZE`: Maximum file size (5 MB default)
- `SEQUENCE_LENGTH`: Number of frames in sequence (5 default)
- `IMAGE_SIZE`: Input image size (224x224 default)

### Model Configuration
The model expects:
- Input: Sequences of 5 images
- Image size: 224x224 pixels
- Input range: 0.0 to 1.0 (normalized)
- Output: Probabilities for 41 action classes

## ⚙️ Model Details

### Architecture
```
Input (Batch, 5, 224, 224, 3)
    ↓
TimeDistributed(MobileNetV2) - Feature extraction
    ↓
TimeDistributed(GlobalAveragePooling2D)
    ↓
LSTM(128) - Temporal modeling
    ↓
Dropout(0.5)
    ↓
Dense(64, ReLU) - Classification
    ↓
Dense(41, Softmax) - Output (41 actions)
```

### Training Details
- Base Model: MobileNetV2 (ImageNet weights)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 8
- Epochs: 25
- Dataset: Custom action videos split into sequences

## 📊 Performance Improvements

If you need to improve model performance:

1. **Increase Training Data**: More diverse examples per action class
2. **Fine-tune MobileNetV2**: Set `cnn.trainable = True` and train with lower learning rate
3. **Longer Sequences**: Increase `SEQUENCE_LENGTH` from 5 to 10-15 frames
4. **Data Augmentation**: Add random crops, rotations, brightness adjustments
5. **Different Base Models**: Try EfficientNet, ResNet50, or Inception
6. **Ensemble Methods**: Combine predictions from multiple models
7. **3D CNNs**: Use 3D convolutions for better temporal feature learning

### Sample Training Code Improvement
```python
# Unfreeze last 20 layers of MobileNetV2
cnn.trainable = True
for layer in cnn.layers[:-20]:
    layer.trainable = False

# Use lower learning rate for fine-tuning
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

## 🐛 Troubleshooting

### Backend won't start
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check if port 5000 is in use: `netstat -ano | findstr :5000` (Windows)
- Try a different port: Change port in `app.py` and update frontend API_BASE_URL

### Model not loading
- Verify `cnn_Lstm_model.h5` exists in the `backend/` directory
- Check file size and integrity
- Error logs in console will show the exact issue

### Frontend can't connect to backend
- Ensure backend is running on port 5000
- Check CORS is enabled (it is in app.py)
- If using different IPs, update `API_BASE_URL` in `static/script.js`

### Memory issues with large images
- Maximum file size is 5 MB by default
- Images are automatically resized to 224x224
- Try uploading smaller/compressed images

### Slow predictions
- GPU inference is faster; ensure CUDA is set up correctly
- Use `tensorflow-gpu` instead of CPU version
- Check system resources (RAM, GPU memory)

## 📚 Training Your Own Model

If you want to train on your own dataset:

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# Similar to your training script
cnn = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
cnn.trainable = False

model = Sequential([
    TimeDistributed(cnn, input_shape=(SEQUENCE_LENGTH,224,224,3)),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS)
model.save("cnn_Lstm_model.h5")
```

## 🌐 Deployment

### Docker (Optional)
Create a `Dockerfile` for containerization:

```dockerfile
FROM python:3.9
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "backend/app.py"]
```

### Cloud Deployment
- **AWS EC2**: Deploy as microservices
- **Google Cloud**: Use Cloud Run for serverless
- **Heroku**: Simple deployment with git push
- **Azure**: Azure App Service

## 📝 File Descriptions

### Backend Files

#### `app.py`
Main Flask backend server with:
- Model loading and inference
- Image preprocessing
- REST API endpoints
- Error handling and logging

#### `requirements.txt`
Python package dependencies:
- Flask 2.3.2 - Web framework
- TensorFlow 2.12.0 - Deep learning
- OpenCV 4.8.0 - Image processing
- NumPy 1.24.3 - Numerical computing
- Pillow 10.0.0 - Image handling

### Frontend Files

#### `app.py`
Simple Flask server to serve HTML templates and static files.

#### `templates/index.html`
Main HTML file with:
- Tab navigation
- Upload interfaces
- Results display
- Information sections

#### `static/style.css`
Comprehensive styling with:
- Responsive design
- Modern UI components
- Smooth animations
- Mobile optimization

#### `static/script.js`
JavaScript for:
- Tab management
- File upload handling
- Drag-and-drop functionality
- API communication
- Results visualization

## 📈 Future Enhancements

1. **Video Input**: Direct video file upload and frame extraction
2. **Real-time Webcam**: Live action recognition from webcam
3. **Batch Processing**: Upload and process multiple sequences
4. **Model Comparison**: Compare predictions from different models
5. **Statistics Dashboard**: Track prediction history and accuracy
6. **User Feedback**: Collect user feedback to improve model
7. **Multi-language Support**: Translate action names to different languages
8. **Model Quantization**: Optimize model for faster inference
9. **Edge Deployment**: Run model on edge devices (Raspberry Pi, mobile)
10. **3D Visualization**: Visualize temporal patterns and attention

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍🏫 Course Context

This application implements concepts learned in your course:
- **Convolutional Neural Networks (CNN)**: Feature extraction from images
- **Recurrent Neural Networks (LSTM)**: Temporal sequence modeling
- **Transfer Learning**: Using pre-trained MobileNetV2
- **Image Classification**: Action recognition from visual data
- **REST APIs**: Backend service communication
- **Web Development**: Frontend user interface

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review error messages in console
3. Ensure all dependencies are installed
4. Verify backend is running before using frontend

## 🎓 Interview Preparation

When explaining this system, cover:
1. **Architecture**: Explain CNN-LSTM combination
2. **Data Flow**: From image upload to prediction
3. **Preprocessing**: Image resizing, normalization
4. **Model**: Layer structure and function
5. **Frontend-Backend Communication**: REST API calls
6. **Performance**: Accuracy metrics and inference time
7. **Improvements**: How to enhance the system
8. **Deployment**: How to put it in production

---

**Good luck with your project! 🚀**
