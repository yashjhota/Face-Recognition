import cv2
import numpy as np
import os
import streamlit as st
import pandas as pd
from datetime import datetime
import uuid

# Initialize attendance log in session state
if 'attendance_log' not in st.session_state:
    st.session_state.attendance_log = []

# Paths
dataset_path = 'Face_Recognition/data/'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# KNN Algorithm
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    test = test / 255.0  # Normalize test data
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1] / 255.0  # Normalize train data
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

# Load Data
def load_data():
    face_data = []
    labels = []
    class_id = 0
    names = {}
    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4]
            data_item = np.load(dataset_path + fx)
            face_data.append(data_item)
            target = class_id * np.ones((data_item.shape[0],))
            class_id += 1
            labels.append(target)
    if len(face_data) == 0:
        return None, None
    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    return trainset, names

# Save Attendance
def save_attendance(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    attendance_log = st.session_state.attendance_log
    if name not in [entry['Name'] for entry in attendance_log]:
        attendance_log.append({'Name': name, 'Time': timestamp, 'Status': '✔️'})
    else:
        for entry in attendance_log:
            if entry['Name'] == name:
                entry['Status'] = '✔️'
    st.session_state.attendance_log = attendance_log

# Streamlit App
st.title("Face Recognition Attendance System")

# Menu
menu = ["Home", "Mark Attendance", "Attendance Log"]
choice = st.sidebar.selectbox("Menu", menu)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if choice == "Home":
    st.subheader("Welcome to the Face Recognition Attendance System! 🚀")
    
    # App Overview
    st.markdown("""
    ### Overview
    The **Face Recognition Attendance System** is a cutting-edge tool for managing attendance efficiently using facial recognition technology.
    - **Mark Attendance**: Real-time facial recognition for marking attendance.
    - **Attendance Log**: View attendance records at any time.
    - **Secure and Reliable**: Your data stays private and secure.
    """)
    
    # How it Works
    st.markdown("""
    ### How It Works:
    1. Navigate to the **Mark Attendance** tab.
    2. Start the camera and position your face for recognition.
    3. The system identifies you and marks your attendance automatically.
    4. View your attendance in the **Attendance Log** tab.
    """)
    
    # Display a motivational quote or fun fact
    st.markdown("### Today's Motivational Quote 🌟")
    st.info("“The only way to do great work is to love what you do.” – Steve Jobs")
    
    # Display statistics or system readiness
    st.markdown("### System Statistics 📊")
    num_users = len(os.listdir(dataset_path))  # Count registered users
    st.metric(label="Registered Users", value=num_users)
    
    # Placeholder for an image or video
    st.image('img.png', caption="Seamless and Intelligent Attendance Tracking")

elif choice == "Mark Attendance":
    st.subheader("Mark Attendance in Real Time")

    trainset, names = load_data()
    if trainset is None:
        st.warning("No face data available. Please train new faces first.")
    else:
        if 'run' not in st.session_state:
            st.session_state.run = False

        if st.button("Start Camera"):
            st.session_state.run = True

        if st.button("Stop Camera"):
            st.session_state.run = False

        if st.session_state.run:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
                for (x, y, w, h) in faces:
                    offset = 10
                    x_start = max(x - offset, 0)
                    y_start = max(y - offset, 0)
                    x_end = min(x + w + offset, frame.shape[1])
                    y_end = min(y + h + offset, frame.shape[0])

                    # Crop the face section
                    face_section = frame[y_start:y_end, x_start:x_end]

                    # Ensure the face_section is valid before resizing
                    if face_section.shape[0] > 0 and face_section.shape[1] > 0:
                        face_section = cv2.resize(face_section, (100, 100))

                        # Use KNN to predict the label
                        out = knn(trainset, face_section.flatten())
                        name = names[int(out)]
                        save_attendance(name)

                        # Draw rectangle and label
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    else:
                        print("Skipped an invalid face_section.")

                stframe.image(frame, channels="BGR")

            cap.release()

elif choice == "Attendance Log":
    st.subheader("Attendance Log")
    if st.session_state.attendance_log:
        df = pd.DataFrame(st.session_state.attendance_log)
        st.table(df)
    else:
        st.write("No attendance marked yet.")
