import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# --- Step 1: Load the WAV file ---
fs, x = wavfile.read('C:/Users/starl/OneDrive/Documents/Praat/hindi_vowel_u')   # Replace 'a.wav' with your own file
if fs != 44100:
    print(f"Warning: File sampling rate is {fs} Hz (expected 44100 Hz)")

# Convert to float & mono (if stereo)
x = x.astype(float)
if x.ndim > 1:
    x = x.mean(axis=1)

# Normalize
x = x / np.max(np.abs(x))

# --- Step 2: Extract a 30 ms voiced frame ---
frame_len = int(0.03 * fs)
start = len(x)//2                # take a middle portion
frame = x[start:start + frame_len] * np.hamming(frame_len)

# --- Step 3: Compute AMDF ---
max_lag = int(0.02 * fs)         # up to 20 ms (≈ 50 Hz lower limit)
amdf = np.zeros(max_lag)
for tau in range(1, max_lag):
    amdf[tau] = np.mean(np.abs(frame[:-tau] - frame[tau:]))

# --- Step 4: Find the first minimum within pitch range ---
min_pitch = int(fs / 400)        # ~400 Hz upper limit
max_pitch = int(fs / 80)         # ~80 Hz lower limit
valid_range = amdf[min_pitch:max_pitch]
tau_min = np.argmin(valid_range) + min_pitch

# --- Step 5: Compute Pitch directly ---
T0 = tau_min / fs
F0 = fs / tau_min

print("=====================================")
print(f"Sampling Frequency  : {fs} Hz")
print(f"Pitch Period (T0)   : {T0*1000:.3f} ms")
print(f"Estimated Pitch (F0): {F0:.2f} Hz")
print("=====================================")

# --- Step 6: Plot AMDF ---
plt.figure(figsize=(8,4))
plt.plot(amdf, label='AMDF curve')
plt.axvline(tau_min, color='r', linestyle='--',
            label=f'Pitch = {F0:.2f} Hz\n(T₀ = {T0*1000:.2f} ms)')
plt.title("Average Magnitude Difference Function (AMDF)")
plt.xlabel("Lag (samples)")
plt.ylabel("AMDF value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
