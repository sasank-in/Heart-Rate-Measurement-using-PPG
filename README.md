# Heart Rate Detection Using Camera

A real-time heart rate monitoring system that uses computer vision to detect heart rate from facial color changes captured by a standard webcam.

## Features

- **Real-time heart rate detection** from webcam feed
- **Enhanced OpenCV-based face detection** with improved 7-point landmarks
- **Fallback support** for basic OpenCV detection
- **Live signal visualization** with frequency analysis
- **User-friendly GUI** with PyQt5
- **Video file support** for offline analysis
- **Enhanced ROI extraction** using precise facial landmarks

## How It Works

The system uses photoplethysmography (PPG) principles:
1. **Face Detection**: Locates face in camera feed using OpenCV
2. **ROI Extraction**: Extracts region of interest from facial area
3. **Color Analysis**: Monitors subtle color changes due to blood flow
4. **Signal Processing**: Applies filtering and FFT analysis
5. **Heart Rate Calculation**: Determines BPM from frequency peaks

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam or camera device

### Install Dependencies

**Minimal Installation (Recommended):**
```bash
pip install -r requirements-core.txt
```

**Full Installation (Maximum Accuracy):**
```bash
pip install -r requirements-full.txt
```

**Manual Installation:**
```bash
pip install numpy opencv-python scipy pyqt5 pyqtgraph
```

### Required Packages
- **OpenCV** (`cv2`) - computer vision and face detection
- **NumPy** - numerical computing
- **SciPy** - signal processing
- **PyQt5** - GUI framework
- **pyqtgraph** - real-time plotting

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
5. **View results** in the heart rate display and signal plots

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
├── face_utilities_enhanced.py # Enhanced OpenCV detection (7 landmarks)
├── face_utilities_opencv.py   # Basic OpenCV face detection (fallback)
├── signal_processing.py       # Signal processing algorithms
├── webcam.py                 # Camera interface
├── video.py                  # Video file interface
├── models/                   # Model files directory
├── requirements-core.txt     # Dependencies
├── requirements-full.txt     # Same as core (simplified)
└── README.md                # This file
```

## Technical Details

### Face Detection
- **Primary**: Enhanced OpenCV detection with 7-point landmarks
- **Fallback**: Basic OpenCV Haar cascades (5-point landmarks)
- **Enhanced ROI extraction** using precise cheek region mapping
- **Improved accuracy** with landmark-based face alignment and better eye detection
- **Real-time performance** optimized for live video with minimal dependencies

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

### Camera Issues
- **Camera not detected**: Check if camera is connected and not used by other apps
- **Permission denied**: Ensure camera permissions are granted
- **Poor image quality**: Check camera drivers and lighting

### Face Detection Issues
- **No face detected**: Improve lighting and face positioning
- **Unstable detection**: Keep head still and ensure clear face view
- **Multiple faces**: System uses the largest detected face

### Heart Rate Issues
- **Erratic readings**: Ensure stable lighting and minimal movement
- **No heart rate**: Wait longer for signal stabilization
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

## Testing

The application can be tested by running the main GUI:
```bash
python run.py
```

## Version History

- **v1.0**: Initial release with dlib dependency
- **v2.0**: OpenCV-based face detection, simplified installation
- **v2.1**: Cleaned codebase, improved stability
- **v3.0**: Enhanced OpenCV detection with improved 7-point landmarks
- **v3.2**: Simplified codebase with reliable OpenCV-only face detection

---

**Note**: This system is for educational and research purposes only. It should not be used for medical diagnosis or health monitoring without proper validation.