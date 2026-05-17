import numpy as np
from scipy.io import wavfile
from PIL import Image as PILImage
from audio_steganography import (
    embed_addition, decode_addition,
    compute_audio_snr, compute_psnr, compute_mse,
)

# ── Load cover audio ──────────────────────────────────────────────────────────
sample_rate, cover_audio = wavfile.read("super duper secret audio.wav")

print(f"dtype: {cover_audio.dtype}")
if cover_audio.dtype == np.int16:
    cover_audio = cover_audio.astype(np.float64) / 32768.0
elif cover_audio.dtype == np.int32:
    cover_audio = cover_audio.astype(np.float64) / 2147483648.0
elif cover_audio.dtype == np.float32:
    cover_audio = cover_audio.astype(np.float64)

print(f"Shape before mono fix: {cover_audio.shape}")
if cover_audio.ndim == 2:
    cover_audio = cover_audio.mean(axis=1)
print(f"Shape after mono fix:  {cover_audio.shape}")

# ── Load secret image (grayscale) ─────────────────────────────────────────────
secret_image = np.array(PILImage.open("9k.png").convert("L"), dtype=np.uint8)
print(f"Secret image shape: {secret_image.shape}")

# ── Embed ─────────────────────────────────────────────────────────────────────
henon_params      = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}
arnold_iterations = 5
alpha             = 0.05

stego_audio = embed_addition(cover_audio, secret_image,
                             alpha=alpha,
                             henon_params=henon_params,
                             arnold_iterations=arnold_iterations)

snr = compute_audio_snr(cover_audio, stego_audio[:len(cover_audio)])
print(f"Audio SNR after embedding: {snr:.2f} dB")

wavfile.write("stego_output_image_addition.wav", sample_rate,
              (stego_audio * 32768).astype(np.int16))
print("Stego audio saved to stego_output_image_addition.wav")

# ── Decode ────────────────────────────────────────────────────────────────────
_, stego_loaded = wavfile.read("stego_output_image_addition.wav")

if stego_loaded.dtype == np.int16:
    stego_loaded = stego_loaded.astype(np.float64) / 32768.0
elif stego_loaded.dtype == np.int32:
    stego_loaded = stego_loaded.astype(np.float64) / 2147483648.0
elif stego_loaded.dtype == np.float32:
    stego_loaded = stego_loaded.astype(np.float64)

if stego_loaded.ndim == 2:
    stego_loaded = stego_loaded.mean(axis=1)

recovered_image = decode_addition(stego_loaded, cover_audio,
                                  henon_params=henon_params,
                                  arnold_iterations=arnold_iterations)

# ── Quality metrics ───────────────────────────────────────────────────────────
# If the image was resized during embedding, compare against the resized version
print(f"Original image shape:  {secret_image.shape}")
print(f"Recovered image shape: {recovered_image.shape}")
if secret_image.shape != recovered_image.shape:
    H, W = recovered_image.shape
    reference = np.array(PILImage.fromarray(secret_image).resize((W, H), PILImage.BILINEAR),
                         dtype=np.uint8)
    print(f"(Original resized to {reference.shape} for fair comparison)")
else:
    reference = secret_image
psnr = compute_psnr(reference, recovered_image)
mse  = compute_mse(reference, recovered_image)
print(f"Image PSNR: {psnr:.2f} dB")
print(f"Image MSE:  {mse:.2f}")

# ── Save recovered image ──────────────────────────────────────────────────────
PILImage.fromarray(recovered_image.astype(np.uint8)).save("recovered_image_addition.png")
print("Recovered image saved to recovered_image_addition.png")
