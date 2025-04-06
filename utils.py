import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class DeepfakeDetectorPredictor:
    def __init__(self, model_path, vectorizer_path, image_size=(224, 224)):
        self.image_size = image_size
        self.model = load_model(model_path)
        self.vectorizer = self._load_vectorizer(vectorizer_path)

    def _load_vectorizer(self, vectorizer_path):
        """Load the vectorizer from a saved JSON file."""
        with open(vectorizer_path, 'r') as f:
            vectorizer_data = json.load(f)

        vectorizer = TfidfVectorizer(max_features=len(vectorizer_data['vocabulary']))
        vectorizer.vocabulary_ = {k: int(v) for k, v in vectorizer_data['vocabulary'].items()}
        vectorizer.idf_ = np.array(vectorizer_data['idf'])
        return vectorizer

    def _process_image(self, image_path):
        """Preprocess the image for prediction."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        return img

    def _extract_metadata(self, image_path):
        """Extract metadata features from the image."""
        metadata = {
            'file_size': os.path.getsize(image_path),
            'image_format': os.path.splitext(image_path)[1].lower(),
        }

        from PIL import Image
        img = Image.open(image_path)
        metadata.update({
            'width': img.size[0],
            'height': img.size[1],
            'mode': img.mode,
        })

        try:
            exif = img._getexif()
            if exif:
                metadata.update({
                    'has_exif': True,
                    'exif_count': len(exif),
                    'software_present': 'Software' in exif,
                    'device_info_present': any(key in exif for key in ['Make', 'Model'])
                })
            else:
                metadata.update({'has_exif': False})
        except:
            metadata.update({'has_exif': False})

        img_array = np.array(img)
        if len(img_array.shape) >= 2:
            metadata.update({
                'mean_pixel_value': float(np.mean(img_array)),
                'std_pixel_value': float(np.std(img_array)),
            })

        return metadata

    def predict(self, image_path):
        """Predict whether the image is genuine or deepfake."""
        try:
            # Process image and metadata
            img = self._process_image(image_path)
            metadata = self._extract_metadata(image_path)
            
            # Prepare metadata for model
            metadata_text = json.dumps(metadata)
            metadata_vector = self.vectorizer.transform([metadata_text]).toarray()
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Predict
            prediction = self.model.predict([img, metadata_vector])
            return "Deepfake" if prediction[0] > 0.5 else "Genuine", prediction[0]
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None