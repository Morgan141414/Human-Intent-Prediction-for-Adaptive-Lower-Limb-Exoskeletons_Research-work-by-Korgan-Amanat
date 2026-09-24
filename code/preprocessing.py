import numpy as np
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Butterworth bandpass filter (zero-phase)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def z_score_normalize(data):
    """Z-score normalization (per-channel)."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1e-6
    return (data - mean) / std

def sliding_window(data, labels, window_size, step_size):
    """Sliding window segmentation."""
    n_samples = data.shape[0]
    windows = []
    window_labels = []
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end, :])
        # Use mode of labels in the window, or just the label at the end
        window_labels.append(labels[end-1])
        
    return np.array(windows), np.array(window_labels)

class IMUDataset(Dataset):
    """PyTorch Dataset for IMU sliding windows."""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def preprocess_subject_data(subject_data, t_onset, horizon_ms, fs=100, window_ms=200, step_ms=10):
    """
    Filter, normalize, and segment data.
    Horizon shifts the label backward relative to T_onset.
    """
    window_size = int(window_ms / 1000.0 * fs)
    step_size = int(step_ms / 1000.0 * fs)
    horizon_samples = int(horizon_ms / 1000.0 * fs)
    
    # Filter
    filtered_data = butter_bandpass_filter(subject_data, 0.5, 20.0, fs)
    
    # Normalize
    normalized_data = z_score_normalize(filtered_data)
    
    # Create labels with prediction horizon
    labels = np.zeros(subject_data.shape[0])
    prediction_target_idx = max(0, t_onset - horizon_samples)
    labels[prediction_target_idx:] = 1 # Simplified to binary intent for demo, normally multi-class
    
    windows, window_labels = sliding_window(normalized_data, labels, window_size, step_size)
    return windows, window_labels
