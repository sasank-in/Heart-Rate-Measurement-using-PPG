import cv2
import numpy as np
import time

class Webcam(object):
    def __init__(self):
        #print ("WebCamEngine init")
        self.dirname = "" #for nothing, just to make 2 inputs the same
        self.cap = None
    
    def start(self):
        print("[INFO] Starting webcam...")
        time.sleep(1)  # wait for camera to be ready
        self.cap = cv2.VideoCapture(0)
        self.valid = False
        
        if not self.cap.isOpened():
            print("[ERROR] Could not open webcam")
            self.shape = None
            return
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.shape = frame.shape
                self.valid = True
                print(f"[INFO] Webcam initialized: {self.shape[1]}x{self.shape[0]}")
            else:
                print("[ERROR] Could not read from webcam")
                self.shape = None
        except Exception as e:
            print(f"[ERROR] Webcam initialization failed: {e}")
            self.shape = None
    
    def get_frame(self):
        if self.valid and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)  # Mirror the image
                return frame
            else:
                self.valid = False
                print("[WARNING] Lost connection to webcam")
        
        # Return error frame if camera not available
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark gray background
        col = (0, 255, 255)  # Yellow text
        cv2.putText(frame, "Camera not accessible", 
                   (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        cv2.putText(frame, "Check camera connection", 
                   (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("[INFO] Stop webcam")
        
