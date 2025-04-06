from flask import Flask, render_template,request,jsonify
from flask_cors import CORS
from utils import DeepfakeDetectorPredictor
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict whether the image is genuine or deepfake."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Save uploaded file
        image = request.files['image']
        os.makedirs('temp',exist_ok=True)
        image_path = os.path.join(app.root_path,'temp', image.filename)
        image.save(image_path)

        detector = DeepfakeDetectorPredictor('model/model.h5','model/vectorizer.json')
        prediction = detector.predict(image_path)
        
        if os.path.exists(image_path):
            os.remove(image_path)
        # Return result
        return jsonify({
            "result": prediction[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)