import cv2
import numpy as np 
import os
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

BASE_PATH = "smart_attendance_system"  
KNOWN_PATH = os.path.join(BASE_PATH, "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_PATH, "attendance.csv")

os.makedirs(KNOWN_PATH, exist_ok=True)

attendance_log = set()

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    if (name, date) not in attendance_log:
        with open(ATTENDANCE_FILE, "a") as f:
            f.write(f"{name},{date},{time}\n")
        attendance_log.add((name, date))

def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir(KNOWN_PATH):
        img_path = os.path.join(KNOWN_PATH, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            known_names.append(os.path.splitext(file)[0])
            known_encodings.append(img)
    return known_encodings, known_names

def train_recognizer():
    known_faces, known_names = load_known_faces()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.array(range(len(known_faces)))
    recognizer.train(known_faces, labels)
    return recognizer, known_names

def add_student_from_file():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    name = simpledialog.askstring("نام دانش‌آموز", "نام را وارد کنید:")
    if not name:
        return
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    save_path = os.path.join(KNOWN_PATH, f"{name}.jpg")
    cv2.imwrite(save_path, img)
    messagebox.showinfo("ثبت شد", f"{name} ذخیره شد.")

def add_student_from_camera():
    name = simpledialog.askstring("نام دانش‌آموز", "نام را وارد کنید:")
    if not name:
        return
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Capture Image", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            save_path = os.path.join(KNOWN_PATH, f"{name}.jpg")
            cv2.imwrite(save_path, face_img)
            messagebox.showinfo("ثبت شد", f"{name} ذخیره شد.")
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_and_save_image(frame, name):
    img_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = os.path.join("captured_faces", img_name)
    os.makedirs("captured_faces", exist_ok=True)
    cv2.imwrite(img_path, frame)
    return img_path

def recognize_faces():
    recognizer, known_names = train_recognizer()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    present_students = set()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            try:
                label, confidence = recognizer.predict(face_img)
                if confidence < 80:
                    name = known_names[label]
                    mark_attendance(name)
                    present_students.add(name)
                    img_path = capture_and_save_image(frame, name)
                    print(f"Image saved at: {img_path}")
                else:
                    name = "Unknown"
            except:
                name = "Error"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    show_attendance_chart(present_students, known_names)

def show_attendance_chart(present_students, all_students):
    absent_students = set(all_students) - present_students
    labels = ['Present', 'Absent']
    sizes = [len(present_students), len(absent_students)]
    colors = ['#4CAF50', '#F44336']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Attendance Summary")
    plt.axis('equal')
    plt.show()

root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("450x300")
root.configure(bg="#f0f4f7")

title = tk.Label(root, text="🎓 Smart Attendance System", font=("Helvetica", 18, "bold"), bg="#f0f4f7", fg="#333")
title.pack(pady=20)

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)

btn1 = ttk.Button(root, text="➕ Add Student (from file)", command=add_student_from_file)
btn1.pack(pady=10)

btn2 = ttk.Button(root, text="📸 Add Student (from camera)", command=add_student_from_camera)
btn2.pack(pady=10)

btn3 = ttk.Button(root, text="✅ Start Attendance", command=recognize_faces)
btn3.pack(pady=10)

footer = tk.Label(root, text="Made by Taha ❤️", font=("Helvetica", 9), bg="#f0f4f7", fg="#888")
footer.pack(side="bottom", pady=10)

root.mainloop()

