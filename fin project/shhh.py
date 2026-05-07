import json
import numpy as np
from scipy.io import wavfile
from audio_steganography import decode_addition

_, cover_audio = wavfile.read("my_jingle.wav")
cover_audio = cover_audio.astype(np.float64) / 32768.0  # normalize to [-1, 1]

# Add this check:
print(f"Shape: {cover_audio.shape}")  # if stereo: (617400, 2) — if mono: (617400,)

# If stereo, convert to mono by averaging the two channels:
if cover_audio.ndim == 2:
    cover_audio = cover_audio.mean(axis=1)

print(f"After fix: {cover_audio.shape}")

_, secret = wavfile.read("stego_output.wav")
secret = secret.astype(np.float64) / 32768.0

henon_params      = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}
arnold_iterations = 5

# Receiver loads it like this
with open("meta.json", "r") as f:
    meta = json.load(f)
    meta["grid_shape"] = tuple(meta["grid_shape"])  

recovered_text = decode_addition(secret, cover_audio, meta,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)
print(recovered_text)    