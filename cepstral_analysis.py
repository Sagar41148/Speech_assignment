import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window, find_peaks
import warnings
import pickle

# -------------------------
# PARAMETERS
# -------------------------
wav_path = 'C:/Users/starl/OneDrive/Documents/Praat/hindi_vowel_uu'     # change to your file name
preemph = 0.97
frame_dur = 0.03              # 30 ms
hop_dur = 0.01                # 10 ms
lifter_ms = 1.5               # ms lifter for smoothing
max_formant_freq = 5000       # Hz
min_pitch = 50                # Hz
max_pitch = 500               # Hz
peak_prominence = 0.1         # relative prominence
min_consecutive_voiced = 6    # per assignment

# -------------------------
# READ AUDIO
# -------------------------
fs, x = wavfile.read(wav_path)
if x.ndim > 1:
    x = np.mean(x, axis=1)  # convert stereo → mono
x = x.astype(float)
x = x / np.max(np.abs(x))   # normalize

# Pre-emphasis
x = np.append(x[0], x[1:] - preemph * x[:-1])

# -------------------------
# FRAMING
# -------------------------
frame_len = int(round(frame_dur * fs))
hop_len = int(round(hop_dur * fs))

num_frames = 1 + (len(x) - frame_len) // hop_len
if num_frames < 1:
    raise ValueError("Audio too short for chosen frame/hop durations.")

frames = np.zeros((frame_len, num_frames))
for i in range(num_frames):
    start = i * hop_len
    frames[:, i] = x[start:start + frame_len]

win = get_window('hamming', frame_len)
frame_times = ((np.arange(num_frames) * hop_len) + frame_len / 2) / fs

# FFT size (4× zero-padding)
nfft = 1
while nfft < frame_len * 4:
    nfft *= 2
M = nfft // 2 + 1
freq_axis = np.linspace(0, fs / 2, M)

# Lifter (convert ms → samples)
lifter_n = int(round(lifter_ms * 1e-3 * fs))

# Preallocations
cepstra = np.zeros((nfft, num_frames))
smoothed_log_spec = np.zeros((M, num_frames))
pitchHz = np.zeros(num_frames)
F = np.full((3, num_frames), np.nan)

# -------------------------
# FRAMEWISE PROCESSING
# -------------------------
for k in range(num_frames):
    fr = frames[:, k] * win
    X = np.fft.rfft(fr, nfft)
    magX = np.abs(X) + np.finfo(float).eps
    logMag = np.log(magX)

    # Real cepstrum
    mirrored = np.concatenate([logMag, logMag[-2:0:-1]])
    c = np.real(np.fft.ifft(mirrored))
    cepstra[:, k] = c

    # Liftering
    cl = c.copy()
    if lifter_n < len(c) // 2:
        cl[lifter_n:len(c) - lifter_n] = 0

    # Smoothed log spectrum
    smLogFull = np.fft.fft(cl)
    smLogPos = np.real(smLogFull[:M])
    smoothed_log_spec[:, k] = smLogPos

    # ---------- FORMANT ESTIMATION ----------
    fidx_max = np.searchsorted(freq_axis, max_formant_freq, side='right') - 1
    rel_prom = peak_prominence * np.max(smLogPos[:fidx_max + 1])
    peaks, _ = find_peaks(smLogPos[:fidx_max + 1], prominence=rel_prom)
    if peaks.size == 0:
        peaks, _ = find_peaks(smLogPos[:fidx_max + 1])
    if peaks.size > 0:
        peak_amps = smLogPos[peaks]
        topN = min(3, len(peak_amps))
        top_idx = np.argsort(peak_amps)[-topN:]
        chosen_freqs = np.sort(freq_axis[peaks[top_idx]])
        F[:len(chosen_freqs), k] = chosen_freqs

    # ---------- PITCH ESTIMATION ----------
    qmin = int(round(fs / max_pitch))
    qmax = int(round(fs / min_pitch))
    qmin = max(qmin, 2)
    qmax = min(qmax, nfft // 2)
    if qmax >= qmin:
        cep_slice = c[qmin:qmax + 1]
        if cep_slice.size > 0:
            mi = np.argmax(cep_slice)
            peak_idx = qmin + mi
            pitch_est = fs / peak_idx
            pitchHz[k] = pitch_est

# -------------------------
# VOICING DECISION
# -------------------------
cep_peak_mag = np.zeros(num_frames)
frame_energy = np.sum(frames ** 2, axis=0)
for k in range(num_frames):
    qmin = int(round(fs / max_pitch))
    qmax = int(round(fs / min_pitch))
    c = cepstra[:, k]
    cep_peak_mag[k] = np.max(c[qmin:qmax + 1])

# Normalize
cep_peak_mag = (cep_peak_mag - np.min(cep_peak_mag)) / (np.ptp(cep_peak_mag) + 1e-8)
frame_energy_norm = (frame_energy - np.min(frame_energy)) / (np.ptp(frame_energy) + 1e-8)

voiced_mask = (cep_peak_mag > 0.25) & (frame_energy_norm > 0.1)

# Find runs of consecutive voiced frames
runs = np.zeros_like(voiced_mask, dtype=int)
run_id = 0
i = 0
while i < len(voiced_mask):
    if voiced_mask[i]:
        run_id += 1
        j = i
        while j < len(voiced_mask) and voiced_mask[j]:
            runs[j] = run_id
            j += 1
        i = j
    else:
        i += 1

unique_runs = [r for r in np.unique(runs) if r != 0]
best_run_len = 0
best_run_id = 0
for rid in unique_runs:
    length = np.sum(runs == rid)
    if length > best_run_len:
        best_run_len = length
        best_run_id = rid

if best_run_len >= min_consecutive_voiced:
    voiced_frames_idx = np.where(runs == best_run_id)[0]
else:
    sorted_energy = np.argsort(frame_energy_norm)[::-1]
    voiced_frames_idx = np.sort(sorted_energy[:min(min_consecutive_voiced, len(sorted_energy))])
    warnings.warn(f"Could not find {min_consecutive_voiced} contiguous voiced frames; using top-energy frames.")

# -------------------------
# COMPUTE AVERAGES
# -------------------------
sel = voiced_frames_idx
F1_mean = np.nanmean(F[0, sel])
F2_mean = np.nanmean(F[1, sel])
F3_mean = np.nanmean(F[2, sel])
valid_pitch_idx = sel[pitchHz[sel] > 0]
pitch_mean = np.mean(pitchHz[valid_pitch_idx]) if valid_pitch_idx.size > 0 else np.nan

print("\nSelected frame indices:", sel.tolist())
print(f"Average F1: {F1_mean:.1f} Hz")
print(f"Average F2: {F2_mean:.1f} Hz")
print(f"Average F3: {F3_mean:.1f} Hz")
print(f"Average Pitch: {pitch_mean:.1f} Hz")

# -------------------------
# PLOTS
# -------------------------
# (1) Cepstrally smoothed log-spectrum
plt.figure(figsize=(9, 4.5))
plt.imshow(smoothed_log_spec, aspect='auto', origin='lower',
           extent=[frame_times[0], frame_times[-1], freq_axis[0], freq_axis[-1]])
plt.colorbar(label='Smoothed log magnitude')
plt.ylim(0, max_formant_freq)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Cepstrally-smoothed log spectrum (frames)')
plt.tight_layout()

# (2) Cepstral sequence (low quefrency)
max_quef_ms = 20
max_quef_idx = min(nfft, int(round(max_quef_ms * 1e-3 * fs)))
q_axis_ms = np.arange(max_quef_idx) / fs * 1000.0
plt.figure(figsize=(9, 4.5))
plt.imshow(cepstra[:max_quef_idx, :], aspect='auto', origin='lower',
           extent=[frame_times[0], frame_times[-1], q_axis_ms[0], q_axis_ms[-1]])
plt.colorbar(label='Cepstrum')
plt.xlabel('Time (s)')
plt.ylabel('Quefrency (ms)')
plt.title('Cepstral sequence (low quefrency region)')
plt.tight_layout()

# (3) Pitch contour & voicing
plt.figure(figsize=(9, 6))
plt.subplot(2, 1, 1)
plt.plot(frame_times, pitchHz, '-o', markersize=3)
plt.plot(frame_times[sel], pitchHz[sel], 'ro', markerfacecolor='r')
plt.xlabel('Time (s)')
plt.ylabel('Pitch (Hz)')
plt.ylim(0, max(500, np.nanmax(pitchHz) + 20))
plt.title('Framewise pitch (cepstral method)')

plt.subplot(2, 1, 2)
plt.plot(frame_times, cep_peak_mag, '-', label='Cepstral peak mag (norm)')
plt.plot(frame_times, frame_energy_norm, '--', label='Energy (norm)')
plt.stem(frame_times, voiced_mask.astype(float), linefmt='k-', markerfmt='k.', basefmt='k-', label='Voiced mask')
plt.xlabel('Time (s)')
plt.ylabel('Normalized')
plt.legend()
plt.title('Voicing decision metrics and mask')
plt.tight_layout()

# (4) Formants over time
plt.figure(figsize=(9, 4.5))
plt.plot(frame_times, F[0, :], 'b.-', label='F1')
plt.plot(frame_times, F[1, :], 'g.-', label='F2')
plt.plot(frame_times, F[2, :], 'r.-', label='F3')
plt.plot(frame_times[sel], F[0, sel], 'bo', markerfacecolor='b')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, max_formant_freq)
plt.legend(loc='upper right')
plt.title('Estimated formant tracks from cepstrally-smoothed spectrum')
plt.grid(True)
plt.tight_layout()

plt.show()

# -------------------------
# SAVE RESULTS
# -------------------------
results = {
    'Ftracks': F,
    'pitchPerFrame': pitchHz,
    'smoothedLogSpec': smoothed_log_spec,
    'cepstra': cepstra,
    'frame_times': frame_times,
    'selected_frames': sel,
    'avgF1': F1_mean,
    'avgF2': F2_mean,
    'avgF3': F3_mean,
    'avgPitch': pitch_mean
}
with open('formant_pitch_results.pkl', 'wb') as fh:
    pickle.dump(results, fh)

print("\nResults saved to formant_pitch_results.pkl")