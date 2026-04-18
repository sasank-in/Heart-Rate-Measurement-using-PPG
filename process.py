import cv2
import numpy as np
import time
from scipy import signal
# Import face utilities - MediaPipe only
try:
    from face_utilities_mediapipe import Face_utilities_mediapipe as Face_utilities, MEDIAPIPE_AVAILABLE
    
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError("MediaPipe failed to load (DLL error)")
    
    DETECTION_METHOD = "mediapipe"
    print("[INFO] Using MediaPipe face detection with age/gender detection")
except ImportError as e:
    print(f"[ERROR] MediaPipe is required but not available: {e}")
    print("\n" + "="*70)
    print("MEDIAPIPE INSTALLATION REQUIRED")
    print("="*70)
    print("\nThis application requires MediaPipe for face detection.")
    print("\nTroubleshooting steps:")
    print("1. Install Visual C++ Redistributable:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("\n2. Reinstall MediaPipe:")
    print("   pip uninstall mediapipe")
    print("   pip install mediapipe==0.10.13")
    print("\n3. If using Python 3.12, try Python 3.11 instead")
    print("   (MediaPipe works best with Python 3.8-3.11)")
    print("\n4. Restart your computer after installing Visual C++")
    print("="*70 + "\n")
    raise ImportError("MediaPipe is required. See troubleshooting steps above.")
from signal_processing import Signal_processing

# Simple rect_to_bb function
def rect_to_bb(rect):
    if hasattr(rect, 'left'):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
    else:
        x, y, w, h = rect
    return (x, y, w, h)

class face_utils:
    @staticmethod
    def rect_to_bb(rect):
        return rect_to_bb(rect)


class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        self.peaks = []
        self.fu = Face_utilities()
        self.sp = Signal_processing()
        
        # Age and gender detection
        self.age = None
        self.gender = None
        
    def extractColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g
        
    def run(self):
        frame = self.frame_in
        
        # Use MediaPipe for face detection with age/gender
        ret_process = self.fu.face_process(frame, "mediapipe")
        if ret_process is None:
            return False
        rects, face, shape, aligned_face, aligned_shape, age, gender = ret_process
        
        # Store age and gender
        self.age = age
        self.gender = gender

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        # Handle MediaPipe landmarks (7 points)
        if aligned_shape is not None and len(aligned_shape) == 7:
            # Enhanced landmarks (7 points) - draw key facial features and ROI regions
            # Draw all landmarks
            for i, (x, y) in enumerate(aligned_shape):
                color = (0, 0, 255) if i < 5 else (255, 0, 0)  # Red for main points, blue for cheeks
                cv2.circle(aligned_face, (x, y), 2, color, -1)
            
            # Draw ROI regions for cheeks
            roi_size = 20
            # Left cheek ROI
            left_cheek = aligned_shape[5]
            cv2.rectangle(aligned_face, 
                        (left_cheek[0] - roi_size, left_cheek[1] - roi_size),
                        (left_cheek[0] + roi_size, left_cheek[1] + roi_size), 
                        (0, 255, 0), 2)
            
            # Right cheek ROI
            right_cheek = aligned_shape[6]
            cv2.rectangle(aligned_face, 
                        (right_cheek[0] - roi_size, right_cheek[1] - roi_size),
                        (right_cheek[0] + roi_size, right_cheek[1] + roi_size), 
                        (0, 255, 0), 2)


        ROIs = self.fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = self.sp.extract_color(ROIs)

        self.frame_out = frame
        self.frame_ROI = aligned_face

        L = len(self.data_buffer)

        g = green_val
        
        if(L > 10 and abs(g-np.mean(self.data_buffer[-10:]))>15): #remove sudden change, if the avg value change is over 15, use the last value
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 10 frames
        if L == self.buffer_size:
            
            self.fps = float(L) / (self.times[-1] - self.times[0])#calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(self.times[0], self.times[-1], L)
            
            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1
            interpolated = np.hamming(L) * interpolated#make the signal become more periodic (advoid spectral leakage)
            #norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated/np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm*30)#do real fft with the normalization multiplied by 10
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
            # idx_remove = np.where((freqs < 50) & (freqs > 180))
            # raw[idx_remove] = 0
            
            self.fft = np.abs(raw)**2#get amplitude spectrum
        
            idx = np.where((freqs > 50) & (freqs < 180))#the range of frequency that HR is supposed to be within 
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq
            self.fft = pruned

            idx2 = np.argmax(pruned)

            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)

            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 3)
        self.samples = processed
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        self.age = None
        self.gender = None
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y 
    

 