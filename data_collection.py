import cv2
import os

# Create a folder to store the images for each person
person_name = input("Enter the person's name: ")
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

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
