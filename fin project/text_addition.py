import numpy as np
from scipy.io import wavfile
from audio_steganography import embed_addition, decode_addition

# Load your audio file
sample_rate, cover_audio = wavfile.read("my_jingle.wav")
cover_audio = cover_audio.astype(np.float64) / 32768.0  # normalize to [-1, 1]

# Add this check:
print(f"Shape: {cover_audio.shape}")  # if stereo: (617400, 2) — if mono: (617400,)

# If stereo, convert to mono by averaging the two channels:
if cover_audio.ndim == 2:
    cover_audio = cover_audio.mean(axis=1)

print(f"After fix: {cover_audio.shape}")

# Secret text
secret_text = "Notice me senpaii"

# Key
henon_params      = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}
arnold_iterations = 5

# Embed
stego_audio, meta = embed_addition(cover_audio, secret_text,
                                   henon_params=henon_params,
                                   arnold_iterations=arnold_iterations)

# Save stego audio
wavfile.write("stego_output.wav", sample_rate,
              (stego_audio * 32768).astype(np.int16))

# --- Later, to decode ---
_, stego_loaded = wavfile.read("stego_output.wav")
stego_loaded = stego_loaded.astype(np.float64) / 32768.0

recovered_text = decode_addition(stego_loaded, cover_audio, meta,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)
print(recovered_text)