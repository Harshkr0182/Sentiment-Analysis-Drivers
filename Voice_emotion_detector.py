import sounddevice as sd
import numpy as np
import librosa
import time

DURATION = 3  # seconds
SAMPLE_RATE = 22050  # standard audio rate

print("🎤 Speak something...")

# Record audio
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

# Convert to 1D array
audio = recording.flatten()

# Extract features
rms = np.mean(librosa.feature.rms(y=audio))
zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

print("\nAnalyzing voice...")
time.sleep(1)

# Simple logic (not ML-based, just simulated emotion)
if rms < 0.01:
    print("🟡 Detected Emotion: Sad/Low")
elif rms < 0.03:
    print("🟢 Detected Emotion: Normal")
else:
    print("🔴 Detected Emotion: Angry/High energy")

print(f"(RMS: {rms:.4f}, ZCR: {zcr:.4f})")
