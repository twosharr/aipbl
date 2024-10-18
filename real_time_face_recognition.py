import cv2
import joblib
import numpy as np
import face_recognition

# Load the trained model and label encoder
classifier = joblib.load('face_recognition_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract face embeddings and recognize faces
def recognize_faces(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        # Predict the name of the person based on the encoding
        y_pred = classifier.predict([face_encoding])
        name = label_encoder.inverse_transform(y_pred)[0]
        names.append(name)

    return face_locations, names

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Unable to access the camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame from webcam.")
            break

        # Recognize faces in the current frame
        face_locations, names = recognize_faces(frame)

        # Draw rectangles around recognized faces
        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Real-Time Face Recognition", frame)

        # Press 'q' to quit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"[ERROR] {str(e)}")

finally:
    # Release the webcam and close the window
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera closed.")
