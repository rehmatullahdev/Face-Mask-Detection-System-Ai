# Face Mask Detection

A Python application that detects whether faces are wearing masks. It provides a Flask web interface and a Tkinter desktop interface, using bundled MediaPipe and Keras models.

## Features

- Real-time webcam mask detection in the web app
- Image upload and analysis
- Desktop image analysis interface
- Detection statistics and saved snapshots
- SQLite logging for detection results
- Admin dashboard for recent detection logs

## Requirements

- Windows, macOS, or Linux
- Python 3.12 recommended
- A working webcam for live detection

The repository includes a local virtual environment in `env/`. For a fresh environment, install the required packages used by the project:

```bash
pip install flask flask-sqlalchemy opencv-python numpy pillow tensorflow mediapipe
```

## Running the Web App

Activate the virtual environment on Windows PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

Start the Flask server:

```bash
python run.py
```

Open [http://localhost:5000](http://localhost:5000) in a browser.

The server runs in debug mode and listens on port `5000` by default.

## Running the Desktop App

With the virtual environment activated, run:

```bash
python desktop_app.py
```

The desktop interface supports selecting an image, running detection, and viewing the results.

## Admin Dashboard

Open [http://localhost:5000/admin/login](http://localhost:5000/admin/login) and sign in with the credentials currently defined by the application:

- Username: `admin`
- Password: `admin`

Change this authentication implementation before deploying the application outside a local development environment.

## Project Structure

```text
run.py                         Flask entry point
desktop_app.py                 Tkinter desktop entry point
app/
  blueprints/                  Web and admin routes
  detectors/                   Face and mask detection components
  resources/models/            Bundled detection models
  services/                    Camera, upload, logging, and pipeline services
  static/uploads/              Generated upload and webcam images
  templates/                   Flask HTML templates
  config.py                    Application configuration
  models.py                    SQLAlchemy models
```

## Generated Files

The application creates these local files and directories at runtime:

- `app/detections.db`: SQLite database
- `app/static/uploads/`: uploaded images and webcam snapshots

They are excluded from Git by `.gitignore`.

## Configuration

Detection and Flask settings are defined in `app/config.py`, including:

- Face confidence threshold: `0.5`
- Mask classification threshold: `0.6`
- Allowed upload extensions
- SQLite database location
- Upload directory

Set `SECRET_KEY` as an environment variable when running outside local development:

```powershell
$env:SECRET_KEY = "replace-with-a-long-random-value"
python run.py
```
