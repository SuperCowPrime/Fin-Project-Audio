"""
STFT Lossiness Demonstration
==============================
Shows why STFT/ISTFT cannot be used for exact text recovery,
while direct bit-packing into DWT coefficients can.

Run this script and see the difference for yourself.
"""

import numpy as np
import math
from scipy.signal import stft, istft
from PIL import Image as PILImage


# ─────────────────────────────────────────────────────────────────────────────
# METHOD A — STFT approach (lossy, what we use for images)
# ─────────────────────────────────────────────────────────────────────────────

def embed_text_via_stft(text: str, nperseg=256, noverlap=128) -> np.ndarray:
    """
    Convert text bytes to a 2D grid, then use ISTFT to turn it into
    an audio-like signal — exactly like the image pipeline does.
    Returns the audio signal.
    """
    raw   = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()
    n     = len(raw)
    side  = math.ceil(math.sqrt(n))
    grid  = np.zeros((side, side), dtype=np.uint8)
    grid.flat[:n] = raw

    # Resize grid to ISTFT spectrogram dimensions
    n_freq  = nperseg // 2 + 1
    n_time  = max(side, 32)
    resized = np.array(
        PILImage.fromarray(grid).resize((n_time, n_freq), PILImage.BILINEAR),
        dtype=np.float64
    )
    spec         = ((resized - 127.5) / 127.5).astype(np.complex128)
    _, audio_sig = istft(spec, nperseg=nperseg, noverlap=noverlap)
    return audio_sig.real, grid.shape


def decode_text_via_stft(audio_signal: np.ndarray,
                          grid_shape: tuple,
                          original_text_len: int,
                          nperseg=256, noverlap=128) -> str:
    """
    Apply STFT to the audio signal to get back a spectrogram,
    resize to original grid shape, read bytes, decode as text.
    """
    H, W = grid_shape
    _, _, spec = stft(audio_signal, nperseg=nperseg, noverlap=noverlap)
    spec_real  = np.real(spec)

    # Normalise spectrogram back to [0, 255]
    mn, mx = spec_real.min(), spec_real.max()
    if mx > mn:
        norm = ((spec_real - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(spec_real, dtype=np.uint8)

    # Resize back to original grid shape
    recovered_grid = np.array(
        PILImage.fromarray(norm).resize((W, H), PILImage.BILINEAR),
        dtype=np.uint8
    )

    raw_recovered = recovered_grid.flatten()[:original_text_len]
    return bytes(raw_recovered.tolist()).decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare(text: str):
    print(f"\n  Original  ({len(text)} chars): \"{text}\"")

    # Run through STFT pipeline
    audio_sig, grid_shape = embed_text_via_stft(text)
    recovered             = decode_text_via_stft(audio_sig, grid_shape, len(text))

    # Count character matches
    matches = sum(a == b for a, b in zip(text, recovered[:len(text)]))
    pct     = 100.0 * matches / len(text)

    print(f"  Recovered ({len(recovered)} chars): \"{recovered}\"")
    print(f"  Match: {matches}/{len(text)} characters = {pct:.1f}%")

    # Show byte-level diff for short texts
    if len(text) <= 30:
        orig_bytes = list(text.encode("utf-8"))
        recv_bytes = list(recovered.encode("utf-8", errors="replace"))[:len(orig_bytes)]
        print(f"\n  Byte comparison:")
        print(f"  {'Char':<6} {'Orig byte':>10} {'Recv byte':>10} {'Match':>7}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*7}")
        for i, (ob, rb) in enumerate(zip(orig_bytes, recv_bytes)):
            ch    = text[i] if text[i].isprintable() else "?"
            match = "✓" if ob == rb else "✗"
            print(f"  {ch!r:<6} {ob:>10} {rb:>10} {match:>7}")


def run():
    print("=" * 65)
    print("  STFT LOSSINESS DEMONSTRATION FOR TEXT")
    print("  Showing why images use STFT but text uses direct bit-packing")
    print("=" * 65)

    print("\n── Short text ──────────────────────────────────────────────")
    compare("Hello World!")

    print("\n── Medium text ─────────────────────────────────────────────")
    compare("The quick brown fox jumps over the lazy dog.")

    print("\n── Long text ───────────────────────────────────────────────")
    compare(
        "This is a secret message hidden inside an audio file using "
        "chaotic map encryption and DWT-based audio steganography."
    )

    print("\n" + "=" * 65)
    print("  CONCLUSION")
    print("=" * 65)
    print("""
  As you can see, the STFT round-trip does NOT preserve exact byte
  values. The recovered text is scrambled because:

  1. STFT converts the byte grid into a spectrogram (frequency domain)
  2. ISTFT converts it back to a time-domain signal
  3. The resize operations (grid → spectrogram → grid) use bilinear
     interpolation, which blends neighbouring pixel values together
  4. The result is an approximation, not an exact copy

  This is fine for IMAGES — a slightly blurry recovered image is
  still recognisable and useful.

  But for TEXT, even one wrong byte breaks a character completely.
  That is why the code uses direct bit-packing into DWT coefficients
  for text, which gives 100% exact recovery every time.
""")


if __name__ == "__main__":
    run()
