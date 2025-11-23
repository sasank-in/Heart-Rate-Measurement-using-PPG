#!/usr/bin/env python3
"""
Face utilities using MediaPipe for superior face detection and landmark tracking.
Includes age and gender detection using OpenCV DNN models.
Provides 468 facial landmarks with high accuracy and real-time performance.
"""

import cv2
import numpy as np
import os

# Import MediaPipe
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"[ERROR] MediaPipe not available: {e}")


class Face_utilities_mediapipe:
    """
    Face utilities using MediaPipe Face Mesh for accurate face detection and landmarks.
    Includes age and gender prediction using pre-trained models.
    """
    
    def __init__(self, face_width=200):
        self.desired_face_width = face_width
        self.desired_face_height = face_width
        self.desired_left_eye = (0.35, 0.35)
        
        # Initialize MediaPipe Face Mesh
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required but not available")
        
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmark indices for face detection
        # MediaPipe provides 468 landmarks, we'll use key points
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398]
        self.NOSE_TIP_INDEX = 1
        self.MOUTH_LEFT_INDEX = 61
        self.MOUTH_RIGHT_INDEX = 291
        self.LEFT_CHEEK_INDEX = 205
        self.RIGHT_CHEEK_INDEX = 425
        
        # Age and Gender detection models
        self.age_net = None
        self.gender_net = None
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        
        # Load age and gender models
        self._load_age_gender_models()
        
        print("[INFO] MediaPipe face detection initialized")
    
    def _load_age_gender_models(self):
        """Load age and gender detection models."""
        try:
            age_proto = 'models/age_deploy.prototxt'
            age_model = 'models/age_net.caffemodel'
            gender_proto = 'models/gender_deploy.prototxt'
            gender_model = 'models/gender_net.caffemodel'
            
            # Check if files exist and are not placeholders (> 1KB)
            age_model_valid = os.path.exists(age_model) and os.path.getsize(age_model) > 1024
            age_proto_valid = os.path.exists(age_proto) and os.path.getsize(age_proto) > 1024
            gender_model_valid = os.path.exists(gender_model) and os.path.getsize(gender_model) > 1024
            gender_proto_valid = os.path.exists(gender_proto) and os.path.getsize(gender_proto) > 1024
            
            if age_proto_valid and age_model_valid:
                self.age_net = cv2.dnn.readNet(age_model, age_proto)
                print("[INFO] Age detection model loaded")
            else:
                print("[WARNING] Age detection models not found or invalid")
                print("[INFO] Age detection will be disabled")
                print("[INFO] To enable: Download models from OpenCV model zoo")
            
            if gender_proto_valid and gender_model_valid:
                self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
                print("[INFO] Gender detection model loaded")
            else:
                print("[WARNING] Gender detection models not found or invalid")
                print("[INFO] Gender detection will be disabled")
                print("[INFO] To enable: Download models from OpenCV model zoo")
                
        except Exception as e:
            print(f"[WARNING] Could not load age/gender models: {e}")
            print("[INFO] Age and gender detection will be disabled")
    
    def predict_age_gender(self, face_roi):
        """
        Predict age and gender from face ROI.
        
        Args:
            face_roi: Face region of interest (BGR image)
            
        Returns:
            tuple: (age_range, gender) or (None, None) if prediction fails
        """
        if self.age_net is None or self.gender_net is None:
            return None, None
        
        if face_roi is None or face_roi.size == 0:
            return None, None
        
        try:
            # Prepare blob for DNN models
            blob = cv2.dnn.blobFromImage(
                face_roi, 
                1.0, 
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            
            # Predict gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            gender_confidence = gender_preds[0].max()
            
            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            age_confidence = age_preds[0].max()
            
            return age, gender
            
        except Exception as e:
            # Silently handle prediction errors
            return None, None
    
    def detect_face_mediapipe(self, frame):
        """
        Detect face using MediaPipe Face Mesh.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (face_rect, landmarks_3d) or (None, None) if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized landmarks to pixel coordinates
        h, w = frame.shape[:2]
        landmarks_px = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_px.append([x, y])
        
        landmarks_px = np.array(landmarks_px)
        
        # Calculate bounding box from landmarks
        x_min = np.min(landmarks_px[:, 0])
        x_max = np.max(landmarks_px[:, 0])
        y_min = np.min(landmarks_px[:, 1])
        y_max = np.max(landmarks_px[:, 1])
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        face_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return face_rect, landmarks_px
    
    def get_key_landmarks(self, landmarks_px, face_rect):
        """
        Extract key facial landmarks for alignment and ROI extraction.
        
        Args:
            landmarks_px: All 468 MediaPipe landmarks in pixel coordinates
            face_rect: Face bounding box (x, y, w, h)
            
        Returns:
            numpy.array: 7 key landmarks (left_eye, right_eye, nose, mouth_left, mouth_right, left_cheek, right_cheek)
        """
        x, y, w, h = face_rect
        
        # Calculate eye centers from multiple eye landmarks
        left_eye_points = landmarks_px[self.LEFT_EYE_INDICES]
        right_eye_points = landmarks_px[self.RIGHT_EYE_INDICES]
        
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        
        # Get other key points
        nose_tip = landmarks_px[self.NOSE_TIP_INDEX]
        mouth_left = landmarks_px[self.MOUTH_LEFT_INDEX]
        mouth_right = landmarks_px[self.MOUTH_RIGHT_INDEX]
        left_cheek = landmarks_px[self.LEFT_CHEEK_INDEX]
        right_cheek = landmarks_px[self.RIGHT_CHEEK_INDEX]
        
        # Adjust coordinates relative to face ROI
        key_landmarks = np.array([
            left_eye_center - [x, y],
            right_eye_center - [x, y],
            nose_tip - [x, y],
            mouth_left - [x, y],
            mouth_right - [x, y],
            left_cheek - [x, y],
            right_cheek - [x, y]
        ])
        
        return key_landmarks
    
    def face_alignment(self, frame, landmarks):
        """
        Align face based on eye positions.
        
        Args:
            frame: Input frame (face ROI)
            landmarks: 7-point key landmarks
            
        Returns:
            tuple: (aligned_face, aligned_landmarks)
        """
        if landmarks is None or len(landmarks) < 2:
            return frame, landmarks
        
        # Get eye centers (first two landmarks)
        left_eye_center = landmarks[0]
        right_eye_center = landmarks[1]
        
        # Calculate angle between eyes
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate desired right eye x-coordinate
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        
        # Calculate distance between eyes
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist if dist > 0 else 1.0
        
        # Calculate center point between eyes
        eyes_center = (int((left_eye_center[0] + right_eye_center[0]) / 2),
                      int((left_eye_center[1] + right_eye_center[1]) / 2))
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update translation component
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        aligned_face = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Transform landmarks
        try:
            landmarks_reshaped = landmarks.reshape(-1, 1, 2).astype(np.float32)
            aligned_landmarks = cv2.transform(landmarks_reshaped, M)
            aligned_landmarks = aligned_landmarks.reshape(-1, 2).astype(np.int32)
        except Exception as e:
            print(f"[DEBUG] Landmark transformation error: {e}")
            aligned_landmarks = landmarks
        
        return aligned_face, aligned_landmarks

    
    def ROI_extraction(self, face, landmarks):
        """
        Extract ROI regions (cheeks) from aligned face using MediaPipe landmarks.
        
        Args:
            face: Aligned face image
            landmarks: 7-point key landmarks
            
        Returns:
            list: List of ROI regions [roi1, roi2] for signal processing
        """
        if landmarks is None or len(landmarks) < 7:
            # Fallback to simple ROI extraction
            h, w = face.shape[:2]
            roi1 = face[int(h*0.4):int(h*0.7), int(w*0.1):int(w*0.4)]
            roi2 = face[int(h*0.4):int(h*0.7), int(w*0.6):int(w*0.9)]
            return [roi1, roi2]
        
        # Use MediaPipe landmarks for precise ROI extraction
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_cheek_point = landmarks[5]
        right_cheek_point = landmarks[6]
        
        # Define cheek regions with optimal size for PPG signal extraction
        roi_size = 40
        
        # Left cheek ROI (from camera perspective - right side of face)
        left_roi_x = max(0, right_cheek_point[0] - roi_size//2)
        left_roi_y = max(0, right_cheek_point[1] - roi_size//2)
        left_roi_x2 = min(face.shape[1], left_roi_x + roi_size)
        left_roi_y2 = min(face.shape[0], left_roi_y + roi_size)
        
        # Right cheek ROI (from camera perspective - left side of face)
        right_roi_x = max(0, left_cheek_point[0] - roi_size//2)
        right_roi_y = max(0, left_cheek_point[1] - roi_size//2)
        right_roi_x2 = min(face.shape[1], right_roi_x + roi_size)
        right_roi_y2 = min(face.shape[0], right_roi_y + roi_size)
        
        roi1 = face[left_roi_y:left_roi_y2, left_roi_x:left_roi_x2]
        roi2 = face[right_roi_y:right_roi_y2, right_roi_x:right_roi_x2]
        
        # Validate ROI sizes
        if roi1.size == 0 or roi2.size == 0:
            # Fallback if ROIs are invalid
            h, w = face.shape[:2]
            roi1 = face[int(h*0.4):int(h*0.7), int(w*0.1):int(w*0.4)]
            roi2 = face[int(h*0.4):int(h*0.7), int(w*0.6):int(w*0.9)]
        
        return [roi1, roi2]
    
    def face_process(self, frame, landmark_type="mediapipe"):
        """
        Process frame for MediaPipe face detection with age/gender detection.
        
        Args:
            frame: Input BGR frame
            landmark_type: Type of landmarks (ignored, always uses MediaPipe)
            
        Returns:
            tuple: (faces, face_roi, landmarks, aligned_face, aligned_landmarks, age, gender) or None
        """
        # Detect face using MediaPipe
        face_rect, all_landmarks = self.detect_face_mediapipe(frame)
        
        if face_rect is None:
            return None
        
        x, y, w, h = face_rect
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
        
        # Get key landmarks relative to face ROI
        key_landmarks = self.get_key_landmarks(all_landmarks, face_rect)
        
        # Align face
        aligned_face, aligned_landmarks = self.face_alignment(face_roi, key_landmarks)
        
        # Predict age and gender
        age, gender = self.predict_age_gender(face_roi)
        
        # Create rectangle object compatible with existing code
        class MediaPipeRect:
            def __init__(self, x, y, w, h):
                self._left = x
                self._top = y
                self._right = x + w
                self._bottom = y + h
                
            def left(self):
                return self._left
                
            def top(self):
                return self._top
                
            def right(self):
                return self._right
                
            def bottom(self):
                return self._bottom
        
        rect_obj = MediaPipeRect(x, y, w, h)
        
        return [rect_obj], face_roi, key_landmarks, aligned_face, aligned_landmarks, age, gender
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Compatibility function
def rect_to_bb(rect):
    """Convert rectangle to bounding box format (x, y, w, h)."""
    if hasattr(rect, 'left'):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
    else:
        x, y, w, h = rect
    
    return (x, y, w, h)
