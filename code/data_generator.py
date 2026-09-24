import numpy as np
import os
import json

def generate_synthetic_data(output_dir="data", n_subjects=12):
    """
    Generate realistic synthetic IMU data for 12 participants and 5 activities.
    Activities: level_walking, stair_ascent, stair_descent, sit_to_stand, turn_90.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    activities = ['level_walking', 'stair_ascent', 'stair_descent', 'sit_to_stand', 'turn_90']
    sensors = ['foot_R', 'shank_R', 'thigh_R', 'pelvis', 'torso']
    channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    fs = 100  # Sampling frequency
    duration = 10  # Seconds per trial
    n_samples = fs * duration
    
    profiles = {}
    
    for subj in range(1, n_subjects + 1):
        subj_id = f"S{subj:02d}"
        profiles[subj_id] = {
            "height": np.round(np.random.normal(1.75, 0.08), 2),
            "weight": np.round(np.random.normal(75, 12), 1),
            "age": int(np.random.normal(35, 10))
        }
        
        subj_dir = os.path.join(output_dir, subj_id)
        os.makedirs(subj_dir, exist_ok=True)
        
        # Inter-subject variability
        amp_scale = np.random.uniform(0.8, 1.2)
        freq_scale = np.random.uniform(0.9, 1.1)
        noise_level = np.random.uniform(0.01, 0.05)
        
        for act in activities:
            # Base signal components
            t = np.linspace(0, duration, n_samples)
            if act == 'level_walking':
                base_freq = 1.0 * freq_scale
            elif act == 'stair_ascent':
                base_freq = 0.8 * freq_scale
            elif act == 'stair_descent':
                base_freq = 0.9 * freq_scale
            else:
                base_freq = 0.3 * freq_scale # Non-cyclic
                
            trial_data = np.zeros((n_samples, len(sensors) * len(channels)))
            
            for s_idx, sensor in enumerate(sensors):
                for c_idx, channel in enumerate(channels):
                    col_idx = s_idx * len(channels) + c_idx
                    
                    # Generate characteristic signal pattern
                    phase = np.random.uniform(0, 2 * np.pi)
                    signal = np.sin(2 * np.pi * base_freq * t + phase)
                    
                    # Add harmonics for complexity
                    signal += 0.5 * np.sin(4 * np.pi * base_freq * t + phase + np.pi/4)
                    
                    # Scale based on sensor location and channel
                    if 'acc' in channel:
                        signal *= 9.81 * amp_scale * (1.0 if 'foot' in sensor else 0.5)
                    else:
                        signal *= 200 * amp_scale * (1.0 if 'foot' in sensor else 0.5)
                        
                    # Add noise
                    signal += np.random.normal(0, noise_level * np.max(np.abs(signal)), n_samples)
                    
                    trial_data[:, col_idx] = signal
                    
            # T_onset marker (onset of intention)
            t_onset = int(0.2 * n_samples) + np.random.randint(-10, 10)
            
            np.save(os.path.join(subj_dir, f"{act}.npy"), trial_data)
            np.save(os.path.join(subj_dir, f"{act}_onset.npy"), np.array([t_onset]))
            
    with open(os.path.join(output_dir, "profiles.json"), "w") as f:
        json.dump(profiles, f, indent=4)
        
    print(f"Generated synthetic data for {n_subjects} subjects in {output_dir}")

if __name__ == "__main__":
    generate_synthetic_data()
