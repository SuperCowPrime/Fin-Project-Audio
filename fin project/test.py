import numpy as np
from scipy.io import wavfile
from audio_steganography import embed_addition, decode_addition

# ── Load cover audio ──────────────────────────────────────────────────────────
sample_rate, cover_audio = wavfile.read("super duper secret audio.wav")

# Check dtype and normalize to [-1, 1]
print(f"dtype: {cover_audio.dtype}")
if cover_audio.dtype == np.int16:
    cover_audio = cover_audio.astype(np.float64) / 32768.0
elif cover_audio.dtype == np.int32:
    cover_audio = cover_audio.astype(np.float64) / 2147483648.0
elif cover_audio.dtype == np.float32:
    cover_audio = cover_audio.astype(np.float64)

# Convert stereo to mono if needed
print(f"Shape before mono fix: {cover_audio.shape}")
if cover_audio.ndim == 2:
    cover_audio = cover_audio.mean(axis=1)
print(f"Shape after mono fix:  {cover_audio.shape}")


henon_params      = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}
arnold_iterations = 5


# ── Decode ────────────────────────────────────────────────────────────────────
_, stego_loaded = wavfile.read("stego_output.wav")

# Apply the same normalization and mono conversion as cover audio
if stego_loaded.dtype == np.int16:
    stego_loaded = stego_loaded.astype(np.float64) / 32768.0
elif stego_loaded.dtype == np.int32:
    stego_loaded = stego_loaded.astype(np.float64) / 2147483648.0
elif stego_loaded.dtype == np.float32:
    stego_loaded = stego_loaded.astype(np.float64)

if stego_loaded.ndim == 2:
    stego_loaded = stego_loaded.mean(axis=1)


recovered_text = decode_addition(stego_loaded, cover_audio,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)

print(f"Recovered: '{recovered_text}'")
