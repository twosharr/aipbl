from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import face_recognition

app = Flask(__name__)

# Load models globally
classifier = None
label_encoder = None

try:
    classifier = joblib.load('face_recognition_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f"[ERROR] Loading model failed: {str(e)}")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Data Collection Route
@app.route('/data_collection', methods=['POST'])
def data_collection():
    person_name = request.form.get('person_name')
    save_path = f"dataset/{person_name}"
    os.makedirs(save_path, exist_ok=True)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    count = 0
    max_images = 100  # Set maximum number of images to capture

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame from webcam.")
            break

        # Display the video feed
        cv2.imshow("Face Data Collection", frame)

        # Save every 10th frame as an image file
        if count % 10 == 0:
            img_name = f"{save_path}/{person_name}_{count}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"[INFO] Saved: {img_name}")

        count += 1

        # Press 'q' to quit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))

# Model Training Route
@app.route('/train_model', methods=['POST'])
def train_model():
    global classifier, label_encoder

    # Prepare training data
    X = []
    y = []
    dataset_dir = "dataset"

    def extract_face_embedding(image):
        face_encodings = face_recognition.face_encodings(image)
        return face_encodings[0] if face_encodings else None

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)

        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(img_path)
                face_embedding = extract_face_embedding(image)

                if face_embedding is not None:
                    X.append(face_embedding)
                    y.append(person_name)

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
    return redirect(url_for('index'))

# Real-Time Face Recognition Route
@app.route('/recognize', methods=['POST'])
def recognize():
    if classifier is None:
        return "Model not loaded", 500  # Handle appropriately

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
        return "Camera access failed", 500

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read frame from webcam.")
                break

            # Convert the frame to RGB and recognize faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            names = []
            for face_encoding in face_encodings:
                y_pred = classifier.predict([face_encoding])
                name = label_encoder.inverse_transform(y_pred)[0]
                names.append(name)

            # Draw rectangles around recognized faces
            for (top, right, bottom, left), name in zip(face_locations, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow("Real-Time Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] {str(e)}")

    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed.")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
