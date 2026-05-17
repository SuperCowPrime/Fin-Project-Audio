import numpy as np
from scipy.io import wavfile
from audio_steganography import embed_override, decode_override

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

# ── Secret text and key ───────────────────────────────────────────────────────
with open("shrek_script.txt", "r", encoding="utf-8") as f:
    secret_text = f.read()
henon_params      = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}
arnold_iterations = 5

# ── Embed ─────────────────────────────────────────────────────────────────────
stego_audio = embed_override(cover_audio, secret_text,
                             henon_params=henon_params,
                             arnold_iterations=arnold_iterations)

# Save stego audio (no meta file needed anymore — everything is inside the audio)
wavfile.write("stego_output_override.wav", sample_rate,
              (stego_audio * 32768).astype(np.int16))
print("Stego audio saved to stego_output_override.wav")

# ── Decode ────────────────────────────────────────────────────────────────────
_, stego_loaded = wavfile.read("stego_output_override.wav")

# Apply the same normalization and mono conversion as cover audio
if stego_loaded.dtype == np.int16:
    stego_loaded = stego_loaded.astype(np.float64) / 32768.0
elif stego_loaded.dtype == np.int32:
    stego_loaded = stego_loaded.astype(np.float64) / 2147483648.0
elif stego_loaded.dtype == np.float32:
    stego_loaded = stego_loaded.astype(np.float64)

if stego_loaded.ndim == 2:
    stego_loaded = stego_loaded.mean(axis=1)


recovered_text = decode_override(stego_loaded,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)

print(f"Recovered: '{recovered_text}'")