import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

skip = 0
face_data = []
dataset_path = 'Face_Recognition/data/'
file_name = input("Enter the name of the person: ")

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])  # Sort by area

    if len(faces) == 0:
        cv2.imshow("Face Section", np.zeros((100, 100, 3), dtype=np.uint8))
        continue

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        skip += 1

        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"Collected {len(face_data)} samples")

        cv2.imshow("Face Section", face_section)

    cv2.imshow("Frame", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert and save face data
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save(dataset_path + file_name + '.npy', face_data)
print(f"Data successfully saved at {dataset_path + file_name}.npy")
print(f"Final shape of data: {face_data.shape}")


cap.release()
cv2.destroyAllWindows()
