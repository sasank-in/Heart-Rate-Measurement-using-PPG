import cv2
import numpy as np
import time
from scipy import signal


class Signal_processing():
    def __init__(self):
        self.a = 1
        
    def extract_color(self, ROIs):
        '''
        extract average value of green color from ROIs
        Enhanced for MediaPipe ROI processing
        '''
        
        g = []
        for ROI in ROIs:
            if ROI is not None and ROI.size > 0:
                # For masked ROIs, only consider non-zero pixels
                if len(ROI.shape) == 3:
                    # Check if this is a masked ROI (has black pixels from masking)
                    non_zero_mask = np.any(ROI != [0, 0, 0], axis=2)
                    if np.any(non_zero_mask):
                        # Extract green channel from non-zero pixels only
                        green_values = ROI[:, :, 1][non_zero_mask]
                        if len(green_values) > 0:
                            g.append(np.mean(green_values))
                    else:
                        # Fallback to simple mean if no masking detected
                        g.append(np.mean(ROI[:, :, 1]))
                else:
                    # Grayscale ROI
                    g.append(np.mean(ROI))
        
        if len(g) == 0:
            return 0  # Return 0 if no valid ROIs found
        
        output_val = np.mean(g)
        return output_val
    
    def normalization(self, data_buffer):
        '''
        normalize the input data buffer
        '''
        
        #normalized_data = (data_buffer - np.mean(data_buffer))/np.std(data_buffer)
        normalized_data = data_buffer/np.linalg.norm(data_buffer)
        
        return normalized_data
    
    def signal_detrending(self, data_buffer):
        '''
        remove overall trending
        
        '''
        detrended_data = signal.detrend(data_buffer)
        
        return detrended_data
        
    def interpolation(self, data_buffer, times):
        '''
        interpolation data buffer to make the signal become more periodic (advoid spectral leakage)
        '''
        L = len(data_buffer)
        
        even_times = np.linspace(times[0], times[-1], L)
        
        interp = np.interp(even_times, times, data_buffer)
        interpolated_data = np.hamming(L) * interp
        return interpolated_data
        
    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        '''
        
        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
        filtered_data = signal.lfilter(b, a, data_buffer)
        
        return filtered_data
        
        