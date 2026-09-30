# 😷 Face Mask Detection System

A **Deep Learning-based Face Mask Detection System** developed as my Final Year Project (FYP). The application detects faces and classifies them as **With Mask** or **Without Mask** using a trained deep learning model.

The system provides a Flask-based web interface for **real-time webcam detection** and **image-based mask detection**, along with detection logging and statistics.

## ✨ Features

- 🎥 Real-time face mask detection using webcam
- 🖼️ Upload images for mask detection
- 👥 Supports detection of multiple faces
- 😷 Classifies faces as **With Mask** or **Without Mask**
- 📊 Detection statistics and results
- 📸 Saves detection snapshots
- 🗃️ SQLite database for detection logs
- 🛡️ Admin dashboard for viewing recent detection records
- 🌐 Flask-based responsive web interface

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **TensorFlow / Keras**
- **MobileNetV3**
- **MediaPipe**
- **OpenCV**
- **SQLite**
- **SQLAlchemy**
- **HTML / CSS / JavaScript**

## 📋 Requirements

- Python 3.10+
- Webcam for real-time detection
- Windows, macOS, or Linux

Install the required dependencies:

```bash
pip install flask flask-sqlalchemy opencv-python numpy pillow tensorflow mediapipe
```

## 🚀 Running the Project

Clone the repository and navigate to the project directory.

Create a virtual environment:

```bash
python -m venv env
```

Activate it on Windows:

```powershell
.\env\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install flask flask-sqlalchemy opencv-python numpy pillow tensorflow mediapipe
```

Start the Flask application:

```bash
python run.py
```

Then open:

```text
http://localhost:5000
```

in your web browser.

## 📁 Project Structure

```text
Face-Mask-Detection/
│
├── run.py                      # Flask application entry point
│
├── app/
│   ├── blueprints/             # Application routes
│   ├── detectors/              # Face and mask detection logic
│   ├── resources/
│   │   └── models/             # Trained AI models
│   ├── services/               # Detection and camera services
│   ├── static/                 # CSS, JavaScript and generated images
│   ├── templates/              # HTML templates
│   ├── config.py               # Application configuration
│   └── models.py               # Database models
│
└── README.md
```

## 🧠 How It Works

1. **MediaPipe** detects faces from webcam frames or uploaded images.
2. The detected face is extracted and preprocessed.
3. The trained **MobileNetV3** model analyzes the face.
4. The system classifies it as:
   - 😷 **With Mask**
   - ❌ **Without Mask**
5. Detection results are displayed through the Flask web interface.
6. Detection information can be stored in the SQLite database for logging and statistics.

## 🎯 Project Purpose

The purpose of this project is to demonstrate the practical application of **Computer Vision, Deep Learning, and Web Development** by integrating a trained face mask classification model into a usable web-based detection system.

## 👨‍💻 Project & Developer

- **Project:** Face Mask Detection System
- **Type:** Final Year Project (FYP)
- **Program:** BS Computer Science
- **University:** Virtual University of Pakistan
- **Developer:** Rehmat Ullah
- **GitHub:** [@rehmatullahdev](https://github.com/rehmatullahdev)

---

⭐ If you find this project useful, consider giving the repository a **star**.