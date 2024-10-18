import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import face_recognition

# Function to extract face embeddings
def extract_face_embedding(image):
    # Get the face embeddings using face_recognition
    face_encodings = face_recognition.face_encodings(image)
    return face_encodings[0] if face_encodings else None

# Prepare training data
X = []
y = []
dataset_dir = "dataset"  # Make sure this directory contains subdirectories for each person

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    
    # Ensure we only process directories
    if os.path.isdir(person_dir):
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            # Load the image file
            image = face_recognition.load_image_file(img_path)
            
            # Extract face embedding
            face_embedding = extract_face_embedding(image)
            
            if face_embedding is not None:  # Check if embedding is valid
                X.append(face_embedding)
                y.append(person_name)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y_encoded)

# Save the model and label encoder
joblib.dump(classifier, 'face_recognition_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model training complete and saved successfully.")
