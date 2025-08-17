# Smart Attendance System - Project Description

## 📌 Introduction
This project is an intelligent attendance system that utilizes facial recognition technology. Developed in Python using OpenCV and Tkinter libraries, it provides an efficient solution for tracking attendance.

## ✨ Key Features
- **Facial Recognition**: Using LBPH (Local Binary Patterns Histograms) algorithm
- **Student Registration**: Via image file or camera capture
- **Attendance Reporting**: Visual pie chart representation of attendance statistics
- **Automatic Saving**: Attendance records stored in CSV format
- **User-Friendly Interface**: Built with Tkinter

## 🛠 Technologies Used
- Python 3
- OpenCV (for image processing and face recognition)
- Tkinter (for GUI)
- Matplotlib (for data visualization)
- NumPy (for numerical computations)

## 📂 Project Structure
```
smart_attendance_system/
├── known_faces/       # Registered face images
├── attendance.csv     # Attendance records
└── main.py            # Main application code
```

## 🚀 Installation & Execution
1. First install required dependencies:
```bash
pip install opencv-python numpy matplotlib
```

2. Run the main application:
```bash
python main.py
```

## 📷 How to Use
1. **Add New Student**:
   - Via image file
   - Via camera capture

2. **Start Attendance**:
   - System automatically detects faces and records attendance

3. **View Report**:
   - Attendance statistics displayed in pie chart format after session

## 💡 Additional Notes
- The system captures and stores images of recognized faces during attendance
- Minimum confidence threshold set to 80% for recognition accuracy
- Developed with ❤️ by Taha

For any issues or contributions, please open an issue or pull request.
