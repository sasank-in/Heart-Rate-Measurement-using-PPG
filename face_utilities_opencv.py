#!/usr/bin/env python3
"""
Alternative face utilities using OpenCV instead of dlib.
This version doesn't require dlib and works with just OpenCV.
"""

import cv2
import numpy as np
from collections import OrderedDict


class Face_utilities_opencv:
    """
    Face utilities using OpenCV's built-in face detection.
    Alternative to dlib-based face detection for easier installation.
    """
    
    def __init__(self, face_width=200):
        self.desired_face_width = face_width
        self.desired_face_height = face_width
        self.desired_left_eye = (0.35, 0.35)
        
        # Load OpenCV's pre-trained face cascade
        self.face_cascade = None
        self.eye_cascade = None
        
        # Try to load cascades
        self._load_cascades()
        
    def _load_cascades(self):
        """Load OpenCV's Haar cascades for face and eye detection."""
        try:
            # Try to load face cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if self.face_cascade.empty() or self.eye_cascade.empty():
                print("[WARNING] Could not load OpenCV cascades")
                return False
            
            print("[INFO] OpenCV face detection initialized")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load cascades: {e}")
            return False
    
    def face_detection(self, frame):
        """
        Detect faces using OpenCV's Haar cascade.
        
        Args:
            frame: Input image frame
            
        Returns:
            list: Detected face rectangles
        """
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve detection in poor lighting
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor for better detection
            minNeighbors=3,    # Fewer neighbors required
            minSize=(50, 50),  # Larger minimum size for more stable detection
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def detect_eyes(self, face_roi):
        """
        Detect eyes in a face region.
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            list: Detected eye rectangles
        """
        if self.eye_cascade is None:
            return []
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10)
        )
        
        return eyes
    
    def get_simple_landmarks(self, face_roi):
        """
        Get simple landmark points using eye detection.
        Returns 5 key points: left eye, right eye, nose tip, left mouth, right mouth.
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            numpy.array: 5 landmark points or None if detection fails
        """
        eyes = self.detect_eyes(face_roi)
        
        if len(eyes) < 2:
            # If we can't detect 2 eyes, create estimated landmarks
            h, w = face_roi.shape[:2]
            landmarks = np.array([
                [int(w * 0.3), int(h * 0.35)],  # Left eye
                [int(w * 0.7), int(h * 0.35)],  # Right eye  
                [int(w * 0.5), int(h * 0.55)],  # Nose tip
                [int(w * 0.35), int(h * 0.75)], # Left mouth corner
                [int(w * 0.65), int(h * 0.75)]  # Right mouth corner
            ])
            return landmarks
        
        # Sort eyes by x-coordinate (left to right)
        eyes = sorted(eyes, key=lambda x: x[0])
        
        # Take the two leftmost eyes as left and right eye
        left_eye = eyes[0]
        right_eye = eyes[1] if len(eyes) > 1 else eyes[0]
        
        # Calculate eye centers
        left_eye_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
        right_eye_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
        
        # Estimate other landmarks based on eye positions
        h, w = face_roi.shape[:2]
        
        # Nose tip (between and below eyes)
        nose_x = (left_eye_center[0] + right_eye_center[0]) // 2
        nose_y = int((left_eye_center[1] + right_eye_center[1]) / 2 + h * 0.2)
        
        # Mouth corners
        mouth_y = int(h * 0.75)
        left_mouth_x = int(left_eye_center[0])
        right_mouth_x = int(right_eye_center[0])
        
        landmarks = np.array([
            left_eye_center,
            right_eye_center,
            [nose_x, nose_y],
            [left_mouth_x, mouth_y],
            [right_mouth_x, mouth_y]
        ])
        
        return landmarks
    
    def face_alignment(self, frame, landmarks):
        """
        Align face based on eye positions.
        
        Args:
            frame: Input frame
            landmarks: 5-point landmarks
            
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
        scale = desired_dist / dist
        
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
            # Return original landmarks if transformation fails
            aligned_landmarks = landmarks
        
        return aligned_face, aligned_landmarks
    
    def ROI_extraction(self, face, landmarks):
        """
        Extract ROI regions (cheeks) from aligned face.
        
        Args:
            face: Aligned face image
            landmarks: 5-point landmarks
            
        Returns:
            tuple: (ROI1, ROI2) - left and right cheek regions
        """
        if landmarks is None or len(landmarks) < 5:
            # Return default ROIs if landmarks not available
            h, w = face.shape[:2]
            roi1 = face[int(h*0.4):int(h*0.7), int(w*0.1):int(w*0.4)]  # Left cheek
            roi2 = face[int(h*0.4):int(h*0.7), int(w*0.6):int(w*0.9)]  # Right cheek
            return roi1, roi2
        
        # Use landmarks to define cheek regions
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        
        # Define cheek regions based on facial landmarks
        # Left cheek (from face perspective)
        roi1_x1 = max(0, right_eye[0] - 20)
        roi1_x2 = min(face.shape[1], right_eye[0] + 30)
        roi1_y1 = max(0, nose[1] - 20)
        roi1_y2 = min(face.shape[0], nose[1] + 30)
        
        # Right cheek (from face perspective)  
        roi2_x1 = max(0, left_eye[0] - 30)
        roi2_x2 = min(face.shape[1], left_eye[0] + 20)
        roi2_y1 = max(0, nose[1] - 20)
        roi2_y2 = min(face.shape[0], nose[1] + 30)
        
        roi1 = face[roi1_y1:roi1_y2, roi1_x1:roi1_x2]
        roi2 = face[roi2_y1:roi2_y2, roi2_x1:roi2_x2]
        
        return roi1, roi2
    
    def no_age_gender_face_process(self, frame, landmark_type="5"):
        """
        Process frame for face detection without age/gender detection.
        
        Args:
            frame: Input frame
            landmark_type: Type of landmarks ("5" for 5-point)
            
        Returns:
            tuple: (faces, face_roi, landmarks, aligned_face, aligned_landmarks) or None
        """
        faces = self.face_detection(frame)
        
        if len(faces) == 0:
            return None
        
        # Take the largest face
        face_rect = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face_rect
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Get landmarks
        landmarks = self.get_simple_landmarks(face_roi)
        
        if landmarks is None:
            return None
        
        # Align face
        aligned_face, aligned_landmarks = self.face_alignment(face_roi, landmarks)
        
        # Convert face_rect to format expected by other parts of the code
        # Create a simple rectangle object that mimics dlib rectangle
        class SimpleRect:
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
        
        rect_obj = SimpleRect(x, y, w, h)
        
        return [rect_obj], face_roi, landmarks, aligned_face, aligned_landmarks


# Compatibility function to convert rectangle format
def rect_to_bb(rect):
    """Convert rectangle to bounding box format (x, y, w, h)."""
    if hasattr(rect, 'left'):
        # dlib-style rectangle
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
    else:
        # OpenCV-style rectangle (already in x, y, w, h format)
        x, y, w, h = rect
    
    return (x, y, w, h)