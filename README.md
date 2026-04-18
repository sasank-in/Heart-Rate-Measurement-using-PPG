# Heart Rate Detection Using Camera

A real-time heart rate monitoring system that uses computer vision to detect heart rate from facial color changes captured by a standard webcam.

## Features

- **Real-time heart rate detection** from webcam feed
- **MediaPipe face detection** with 468 high-accuracy facial landmarks
- **Age and gender detection** using deep learning models
- **Live signal visualization** with frequency analysis
- **User-friendly GUI** with PyQt5
- **Video file support** for offline analysis
- **Precise ROI extraction** using MediaPipe facial landmarks

## How It Works

The system uses photoplethysmography (PPG) principles:
1. **Face Detection**: Locates face in camera feed using OpenCV
2. **ROI Extraction**: Extracts region of interest from facial area
3. **Color Analysis**: Monitors subtle color changes due to blood flow
4. **Signal Processing**: Applies filtering and FFT analysis
5. **Heart Rate Calculation**: Determines BPM from frequency peaks

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download age/gender models (first time only)
python download_age_gender_models.py

# Run application
python run.py
```

## Installation

### Prerequisites
- Python 3.7 or higher (3.8-3.11 recommended for MediaPipe)
- Webcam or camera device

### Install Dependencies

**Recommended Installation:**
```bash
pip install -r requirements.txt
```

**Manual Installation:**
```bash
pip install numpy opencv-python scipy pyqt5 pyqtgraph "mediapipe==0.10.14" "protobuf>=4.21,<5"
```

### Required Packages
- **MediaPipe** `==0.10.14` — face detection with 468 landmarks. Newer 0.10.x releases drop `mp.solutions` on Python 3.12 and will fail at startup
- **protobuf** `>=4.21,<5` — protobuf 5+ breaks MediaPipe 0.10.14's FaceMesh init
- **OpenCV** (`cv2`) — computer vision and age/gender detection models
- **NumPy** — numerical computing
- **SciPy** — signal processing
- **PyQt5** — GUI framework
- **pyqtgraph** — real-time plotting

## Usage

### Start the Application
```bash
python run.py
```

### Using the GUI
1. **Launch** the application
2. **Click "Start"** to begin heart rate detection
3. **Position your face** in the camera view with good lighting
4. **Wait 10-15 seconds** for measurements to stabilize
5. **View results** including:
   - Real-time heart rate (BPM)
   - Age range estimation
   - Gender detection
   - Signal and FFT plots

### Tips for Best Results
- **Good lighting**: Ensure your face is well-lit
- **Stable position**: Keep your head relatively still
- **Clear view**: Make sure your face is clearly visible
- **Avoid movement**: Minimize head movements during measurement
- **Wait for stabilization**: Allow 10-15 seconds for accurate readings

## Project Structure

```
├── run.py                      # Main GUI application
├── process.py                 # Heart rate processing logic
├── face_utilities_mediapipe.py # MediaPipe face detection with age/gender
├── signal_processing.py       # Signal processing algorithms
├── webcam.py                 # Camera interface
├── video.py                  # Video file interface
├── models/                   # Age and gender detection models
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Technical Details

### Face Detection
- **MediaPipe Face Mesh** with 468 high-accuracy landmarks
- **ML-based detection** trained on diverse datasets
- **Precise ROI extraction** using cheek region landmarks (indices 205, 425)
- **Real-time performance** optimized for live video processing
- **Robust tracking** handles head rotation and varying lighting

### Age and Gender Detection
- **Deep learning models** using OpenCV DNN
- **Age ranges**: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- **Gender classification**: Male/Female
- **Real-time prediction** from detected face region

### Signal Processing
- Bandpass filtering (0.8-3 Hz for heart rate range)
- Signal detrending and normalization
- FFT analysis for frequency domain processing
- Peak detection for BPM calculation

### Heart Rate Range
- Typical range: 50-180 BPM
- Frequency range: 0.8-3 Hz
- Buffer size: 100 samples for stability

## Troubleshooting

### MediaPipe Installation Issues

**"MediaPipe is required but not available" / `module 'mediapipe' has no attribute 'solutions'`:**
You installed a newer MediaPipe (e.g. 0.10.33) that doesn't expose `mp.solutions` on Python 3.12. Pin the version:
```bash
pip install "mediapipe==0.10.14"
```

**`AttributeError: 'FieldDescriptor' object has no attribute 'label'`:**
protobuf is too new for MediaPipe 0.10.14. Downgrade:
```bash
pip install "protobuf>=4.21,<5"
```

**"DLL load failed" errors on Windows:**
- Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Restart your computer after installation
- Python 3.8–3.12 supported (3.8–3.11 smoothest; 3.12 works with the pinned versions above)

### Age/Gender Models
If age/gender detection doesn't work:
- Run: `python download_age_gender_models.py`
- Models will be downloaded to the `models/` folder
- Restart the application

### Camera Issues
- **Camera not detected**: Check if camera is connected and not used by other apps
- **Permission denied**: Ensure camera permissions are granted
- **Poor image quality**: Check camera drivers and lighting

### Face Detection Issues
- **No face detected**: Improve lighting and face positioning
- **Unstable detection**: Keep head still and ensure clear face view
- **Age/Gender not showing**: Wait a few seconds for detection to stabilize

### Heart Rate Issues
- **Erratic readings**: Ensure stable lighting and minimal movement
- **No heart rate**: Wait longer for signal stabilization (10-15 seconds)
- **Inaccurate readings**: Check lighting conditions and face positioning

## System Requirements

### Minimum Requirements
- **OS**: Windows 7+, macOS 10.12+, Linux
- **Python**: 3.7+
- **RAM**: 4GB
- **Camera**: Any USB webcam or built-in camera

### Recommended Requirements
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8+
- **RAM**: 8GB
- **Camera**: HD webcam with good low-light performance

## Limitations

- **Lighting dependent**: Requires adequate lighting for accurate detection
- **Movement sensitive**: Head movement can affect accuracy
- **Individual variation**: Accuracy may vary between individuals
- **Not medical grade**: For educational/research purposes only

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Not intended for medical diagnosis.

## Acknowledgments

- OpenCV community for computer vision tools
- PyQt5 for GUI framework
- Scientific Python community for signal processing libraries

## Running the Application

```bash
python run.py
```

Expected console output:
```
[INFO] Using MediaPipe face detection with age/gender detection
[INFO] Age detection model loaded
[INFO] Gender detection model loaded
[INFO] MediaPipe face detection initialized
```

Note: You may see harmless warnings from TensorFlow/MediaPipe - these can be ignored.

## Documentation

- **[USAGE.md](USAGE.md)** - Detailed usage instructions
- **[MEDIAPIPE_TROUBLESHOOTING.md](MEDIAPIPE_TROUBLESHOOTING.md)** - MediaPipe installation help

## Version

**Current Version**: 5.0

- MediaPipe face detection (468 landmarks)
- Real-time heart rate monitoring
- Age and gender detection
- Signal visualization

---

**Note**: This system is for educational and research purposes only. It should not be used for medical diagnosis or health monitoring without proper validation.