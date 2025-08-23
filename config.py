#!/usr/bin/env python3
"""
Configuration settings for Heart Rate Detection system.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ProcessingConfig:
    """Configuration for signal processing parameters."""
    
    # Buffer settings
    buffer_size: int = 100
    min_buffer_size: int = 50
    
    # Heart rate range (BPM)
    min_bpm: float = 48.0
    max_bpm: float = 180.0
    
    # Frequency range (Hz)
    min_freq: float = 0.8  # 48 BPM
    max_freq: float = 3.0  # 180 BPM
    
    # Filter settings
    bandpass_order: int = 5
    outlier_threshold: float = 15.0  # Threshold for outlier detection
    
    # Stability detection
    stability_window: int = 50  # Frames to check for stability
    stability_threshold: float = 5.0  # BPM variation threshold
    
    # FFT settings
    fft_amplification: float = 30.0


@dataclass
class FaceDetectionConfig:
    """Configuration for face detection and processing."""
    
    # Face alignment settings
    desired_face_width: int = 200
    desired_left_eye: Tuple[float, float] = (0.35, 0.35)
    
    # Landmark model paths
    landmarks_68_path: str = "shape_predictor_68_face_landmarks.dat"
    landmarks_5_path: str = "shape_predictor_5_face_landmarks.dat"
    
    # Detection settings
    detection_confidence: float = 0.5
    use_5_point_landmarks: bool = True  # Use 5-point for better performance


@dataclass
class CameraConfig:
    """Configuration for camera and video input."""
    
    # Camera settings
    camera_index: int = 0
    default_fps: float = 30.0
    
    # Frame processing
    frame_width: int = 640
    frame_height: int = 480
    
    # Video settings
    video_resize_width: int = 640


@dataclass
class GUIConfig:
    """Configuration for GUI interface."""
    
    # Window settings
    window_width: int = 1160
    window_height: int = 640
    
    # Display settings
    display_width: int = 640
    display_height: int = 480
    roi_width: int = 200
    roi_height: int = 200
    
    # Plot settings
    plot_width: int = 480
    plot_height: int = 192
    plot_update_interval: int = 200  # milliseconds
    
    # Font settings
    font_size: int = 16


@dataclass
class ColorAmplificationConfig:
    """Configuration for Eulerian Video Magnification."""
    
    # Amplification settings
    amplification_factor: float = 30.0
    pyramid_levels: int = 3
    
    # Frequency range for color amplification
    low_freq: float = 0.4
    high_freq: float = 2.0


class Config:
    """Main configuration class that combines all settings."""
    
    def __init__(self):
        self.processing = ProcessingConfig()
        self.face_detection = FaceDetectionConfig()
        self.camera = CameraConfig()
        self.gui = GUIConfig()
        self.color_amplification = ColorAmplificationConfig()
        
        # Validate model file paths
        self._validate_model_paths()
    
    def _validate_model_paths(self):
        """Validate that required model files exist."""
        models = [
            self.face_detection.landmarks_68_path,
            self.face_detection.landmarks_5_path
        ]
        
        missing_models = []
        for model_path in models:
            if not os.path.exists(model_path):
                missing_models.append(model_path)
        
        if missing_models:
            print(f"[WARNING] Missing model files: {missing_models}")
            print("Please download the dlib facial landmark models:")
            print("- shape_predictor_68_face_landmarks.dat")
            print("- shape_predictor_5_face_landmarks.dat")
    
    def get_frequency_range_hz(self) -> Tuple[float, float]:
        """Get frequency range in Hz."""
        return (self.processing.min_freq, self.processing.max_freq)
    
    def get_bpm_range(self) -> Tuple[float, float]:
        """Get BPM range."""
        return (self.processing.min_bpm, self.processing.max_bpm)
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'processing': self.processing.__dict__,
            'face_detection': self.face_detection.__dict__,
            'camera': self.camera.__dict__,
            'gui': self.gui.__dict__,
            'color_amplification': self.color_amplification.__dict__
        }


# Global configuration instance
config = Config()


def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_file: Path to configuration file (JSON format)
        
    Returns:
        Config: Configuration instance
    """
    if config_file and os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            config.update_from_dict(config_dict)
            print(f"[INFO] Loaded configuration from {config_file}")
        except Exception as e:
            print(f"[WARNING] Failed to load config file {config_file}: {e}")
            print("[INFO] Using default configuration")
    
    return config


def save_config(config_obj: Config, config_file: str):
    """
    Save configuration to file.
    
    Args:
        config_obj: Configuration instance to save
        config_file: Path to save configuration file
    """
    try:
        import json
        with open(config_file, 'w') as f:
            json.dump(config_obj.to_dict(), f, indent=2)
        print(f"[INFO] Configuration saved to {config_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save configuration: {e}")


if __name__ == "__main__":
    # Example usage
    cfg = load_config()
    print("Current configuration:")
    print(f"Buffer size: {cfg.processing.buffer_size}")
    print(f"BPM range: {cfg.get_bpm_range()}")
    print(f"Face width: {cfg.face_detection.desired_face_width}")
    
    # Save example configuration
    save_config(cfg, "config_example.json")