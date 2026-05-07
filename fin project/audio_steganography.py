"""
Audio Steganography with Chaotic Encryption
============================================
Based on: Nasr et al. (2024) - Scientific Reports, 14:22054

Supports TWO secret payload types:
  ─ IMAGE : a grayscale image (2D uint8 numpy array)
  ─ TEXT  : any UTF-8 string

Both types are encrypted with the same three chaotic maps before hiding.

Two embedding strategies (choose based on your trade-off):
  ─ ADDITION  : secret is ADDED to the high-frequency DWT band.
                → Best audio quality (high SNR).
                → Requires original cover audio to decode.
  ─ OVERRIDE  : secret REPLACES the high-frequency DWT band entirely.
                → No original audio needed to decode.
                → Slightly lower audio quality.

How image vs text are embedded differently:
  Images use STFT-based conversion (image ↔ spectrogram ↔ audio signal).
  Text uses direct bit-packing into DWT coefficients (lossless, exact).
  This distinction matters because STFT is a lossy transform — fine for
  approximate image recovery, but would corrupt the exact bytes of text.

Dependencies:
    pip install numpy scipy Pillow PyWavelets
"""

import numpy as np
import pywt
import struct
from scipy.signal import stft, istft
from PIL import Image as PILImage
import math


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CHAOTIC MAP ENCRYPTION
# ─────────────────────────────────────────────────────────────────────────────

def henon_encrypt(grid: np.ndarray, x0=0.01, x1=0.02, a=0.3, b=1.4) -> np.ndarray:
    """
    Henon chaotic map encryption — XORs every element with a chaotic sequence.

    Generates the sequence using: x[i+2] = 1 - a·x[i+1]² + b·x[i]
    then converts to integers [0,255] and XORs with the data.

    XOR is self-inverse: encrypt(encrypt(x)) == x, so decrypt == encrypt.

    Works on any 2D uint8 array (image pixels, text bytes packed into a grid).
    Key parameters: x0, x1, a, b
    """
    flat = grid.flatten().astype(np.int32)
    n    = len(flat)

    x    = np.zeros(n + 2)
    x[0] = x0
    x[1] = x1
    for i in range(n):
        val      = 1.0 - a * (x[i + 1] ** 2) + b * x[i]
        x[i + 2] = float(np.clip(val, -1e6, 1e6))

    chaotic_int = (np.abs(x[2:]) * 1000).astype(np.int64) % 256
    return (flat ^ chaotic_int).astype(np.uint8).reshape(grid.shape)


def henon_decrypt(grid: np.ndarray, x0=0.01, x1=0.02, a=0.3, b=1.4) -> np.ndarray:
    """XOR is self-inverse — decryption is identical to encryption."""
    return henon_encrypt(grid, x0, x1, a, b)


def arnold_cat_map(grid: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Arnold Cat Map — shuffles POSITIONS of elements (values unchanged).
    Transformation per iteration: (x,y) → ((x+y) mod N, (x+2y) mod N)
    Requires a square N×N grid.
    Key parameter: iterations
    """
    N      = grid.shape[0]
    assert grid.shape[0] == grid.shape[1], "Arnold Cat Map requires a square grid"
    result = grid.copy()
    for _ in range(iterations):
        new_grid = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                new_grid[(x + y) % N, (x + 2 * y) % N] = result[x, y]
        result = new_grid
    return result


def arnold_cat_map_inverse(grid: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Inverse Arnold Cat Map.
    Inverse transformation: (x,y) → ((2x-y) mod N, (-x+y) mod N)
    """
    N      = grid.shape[0]
    assert grid.shape[0] == grid.shape[1], "Arnold Cat Map requires a square grid"
    result = grid.copy()
    for _ in range(iterations):
        new_grid = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                new_grid[(2 * x - y) % N, (-x + y) % N] = result[x, y]
        result = new_grid
    return result


def _make_baker_key(N: int) -> list:
    """
    Auto-generate a valid Baker key: equal-width strips summing to N.
    Uses the largest divisor of N that is between 2 and 4 (inclusive),
    guaranteeing equal strip widths so the inverse map is always exact.
    """
    for n_strips in [4, 3, 2]:
        if N % n_strips == 0:
            strip = N // n_strips
            return [strip] * n_strips
    # N is prime or 1 — use a single strip (no permutation)
    return [N]


def baker_map(grid: np.ndarray, key: list) -> np.ndarray:
    """
    Chaotic Baker Map — rearranges vertical strips of the grid.
    key = [n1, n2, ...] where sum(key) == N.
    """
    N      = grid.shape[0]
    assert sum(key) == N
    flat   = grid.flatten()
    result = np.zeros_like(flat)
    cum    = [0]
    for ni in key:
        cum.append(cum[-1] + ni)
    for k, ni in enumerate(key):
        Ni = cum[k]
        for r in range(Ni, Ni + ni):
            for s in range(N):
                nr = min(int((N / ni) * (r - Ni) + s % (N // ni)), N - 1)
                ns = min(int((ni / N) * (s - s % (N // ni)) + Ni),   N - 1)
                result[nr * N + ns] = flat[r * N + s]
    return result.reshape(grid.shape)


def baker_map_inverse(grid: np.ndarray, key: list) -> np.ndarray:
    """Inverse Baker Map."""
    N      = grid.shape[0]
    assert sum(key) == N
    flat   = grid.flatten()
    result = np.zeros_like(flat)
    cum    = [0]
    for ni in key:
        cum.append(cum[-1] + ni)
    for k, ni in enumerate(key):
        Ni = cum[k]
        for r in range(Ni, Ni + ni):
            for s in range(N):
                nr = min(int((N / ni) * (r - Ni) + s % (N // ni)), N - 1)
                ns = min(int((ni / N) * (s - s % (N // ni)) + Ni),   N - 1)
                result[r * N + s] = flat[nr * N + ns]
    return result.reshape(grid.shape)


def encrypt_grid(grid: np.ndarray, henon_params: dict,
                 arnold_iterations: int) -> np.ndarray:
    """
    Apply all three chaotic maps in sequence:
        Henon XOR  →  Arnold position shuffle  →  Baker strip shuffle

    Baker key is auto-generated from grid side length.
    Grid must be square (guaranteed by pad_to_square for text,
    and enforced for images by making them square before calling).
    """
    N   = grid.shape[0]
    key = _make_baker_key(N)
    out = henon_encrypt(grid, **henon_params)
    out = arnold_cat_map(out, arnold_iterations)
    out = baker_map(out, key)
    return out


def decrypt_grid(grid: np.ndarray, henon_params: dict,
                 arnold_iterations: int) -> np.ndarray:
    """Reverse all three maps: Baker⁻¹ → Arnold⁻¹ → Henon XOR."""
    N   = grid.shape[0]
    key = _make_baker_key(N)
    out = baker_map_inverse(grid, key)
    out = arnold_cat_map_inverse(out, arnold_iterations)
    out = henon_decrypt(out, **henon_params)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HELPERS FOR PACKING DATA INTO SQUARE GRIDS
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_square(arr: np.ndarray) -> tuple:
    """
    Pad a 1D uint8 array with zeros to fill the smallest square grid that fits.
    Returns (2D square grid, original_length).
    Arnold and Baker maps require square grids.
    """
    n    = len(arr)
    side = math.ceil(math.sqrt(n))
    buf  = np.zeros(side * side, dtype=np.uint8)
    buf[:n] = arr
    return buf.reshape(side, side), n


def image_to_square(image: np.ndarray) -> tuple:
    """
    Resize/pad a grayscale image to square so it works with Arnold/Baker maps.
    Returns (square grid, original_shape).
    """
    H, W   = image.shape
    side   = max(H, W)
    square = np.zeros((side, side), dtype=np.uint8)
    square[:H, :W] = image
    return square, (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — IMAGE ↔ AUDIO CONVERSION  (STFT / ISTFT — approximate)
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_audio_signal(grid: np.ndarray,
                          nperseg: int = 256, noverlap: int = 128) -> np.ndarray:
    """
    Convert a 2D uint8 grid to a 1D audio-like signal using ISTFT.

    The grid is resized to match the ISTFT spectrogram dimensions
    (n_freq = nperseg//2+1  frequency bins), then passed through ISTFT.
    This is used for IMAGE payloads only.

    NOTE: This is a LOSSY transform — suitable for images (approximate
    recovery is acceptable) but NOT for text (which requires exact bytes).
    """
    n_freq  = nperseg // 2 + 1
    n_time  = max(grid.shape[1], 32)
    resized = np.array(
        PILImage.fromarray(grid.astype(np.uint8)).resize(
            (n_time, n_freq), PILImage.BILINEAR),
        dtype=np.float64
    )
    spec         = ((resized - 127.5) / 127.5).astype(np.complex128)
    _, audio_sig = istft(spec, nperseg=nperseg, noverlap=noverlap)
    return audio_sig.real


def audio_signal_to_grid(audio_signal: np.ndarray,
                          grid_shape: tuple,
                          nperseg: int = 256, noverlap: int = 128) -> np.ndarray:
    """
    Convert a 1D audio signal back to a 2D uint8 grid using STFT.
    Inverse of grid_to_audio_signal().
    """
    H, W = grid_shape
    _, _, spec = stft(audio_signal, nperseg=nperseg, noverlap=noverlap)
    sr = np.real(spec)
    mn, mx = sr.min(), sr.max()
    norm = ((sr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn \
           else np.zeros_like(sr, dtype=np.uint8)
    return np.array(
        PILImage.fromarray(norm).resize((W, H), PILImage.BILINEAR),
        dtype=np.uint8
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TEXT ↔ DWT BITS  (exact, lossless)
# ─────────────────────────────────────────────────────────────────────────────

# We skip the STFT conversion for text and instead directly encode each bit
# of the (encrypted) text into a small ±scale perturbation of the DWT
# high-frequency coefficients.  On decode we compare coefficients to the
# original (addition method) or use sign alone (override method) to recover
# each bit, then unpack bytes.

BIT_SCALE  = 0.0005   # strength of per-bit perturbation (±scale)
BIT_OFFSET = 10     # skip first N coefficients (wavelet boundary effects)

# ─────────────────────────────────────────────────────────────────────────────
# SELF-DESCRIBING HEADER — embedded into the DWT band before the payload
# ─────────────────────────────────────────────────────────────────────────────
#
# The header stores all meta information needed to decode, so the receiver
# does NOT need to pass a meta dict — they just call decode() and get the result.
#
# Header layout (17 bytes = 136 bits), big-endian:
#   B  (1 byte)  : mode         — 0=text, 1=image
#   I  (4 bytes) : payload_len  — upgraded from H(2) to I(4): max 4,294,967,295 bytes
#   I  (4 bytes) : grid_side    — upgraded from H(2) to I(4): supports huge grids
#   I  (4 bytes) : img_H        — original image height (0 for text)
#   I  (4 bytes) : img_W        — original image width  (0 for text)
#   I  (4 bytes) : n_bits       — number of payload bits embedded after header
#   H  (2 bytes) : coeff_scale  — override image scale * 1000 (0 otherwise)
#   h  (2 bytes) : alpha        — addition image alpha * 1000
#
# Total: 1+4+4+4+4+4+2+2 = 25 bytes = 200 bits

HEADER_FORMAT = ">BIIIIIHh"
HEADER_BYTES  = struct.calcsize(HEADER_FORMAT)   # = 25
HEADER_BITS   = HEADER_BYTES * 8                 # = 200


def _pack_header(mode: int, payload_len: int, grid_side: int,
                 img_H: int, img_W: int, n_bits: int,
                 coeff_scale: float = 0.0, alpha: float = 0.05) -> np.ndarray:
    """
    Pack meta information into a 25-byte binary header.
    payload_len and grid_side are now 32-bit (I), supporting payloads
    up to ~4 GB instead of the old 65,535-byte limit.
    Returns a uint8 numpy array of length HEADER_BYTES.
    """
    cs_int = int(round(coeff_scale * 1000))
    al_int = int(round(alpha       * 1000))
    raw    = struct.pack(HEADER_FORMAT, mode, payload_len, grid_side,
                         img_H, img_W, n_bits, cs_int, al_int)
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _unpack_header(header_bytes: np.ndarray) -> dict:
    """
    Unpack a 25-byte header array back into a meta dict.
    """
    raw  = bytes(header_bytes[:HEADER_BYTES].tolist())
    mode, payload_len, grid_side, img_H, img_W, n_bits, cs_int, al_int = \
        struct.unpack(HEADER_FORMAT, raw)
    return {
        "mode"        : "text"  if mode == 0 else "image",
        "payload_len" : int(payload_len),
        "grid_shape"  : (int(grid_side), int(grid_side)),
        "orig_shape"  : (int(img_H), int(img_W)),
        "n_bits"      : int(n_bits),
        "coeff_scale" : cs_int / 1000.0,
        "alpha"       : al_int / 1000.0,
    }


def _embed_header_bits(hf_band: np.ndarray, header_arr: np.ndarray) -> np.ndarray:
    """
    Write header bytes into the DWT band using the same ±BIT_SCALE encoding
    as the payload, starting at BIT_OFFSET.
    Header occupies positions BIT_OFFSET … BIT_OFFSET + HEADER_BITS - 1.
    """
    result   = hf_band.copy()
    hdr_bits = np.unpackbits(header_arr)
    for i, bit in enumerate(hdr_bits):
        idx         = BIT_OFFSET + i
        result[idx] = hf_band[idx] + BIT_SCALE * (1 if bit else -1)
    return result


def _decode_header_bits_addition(hf_stego: np.ndarray,
                                  hf_original: np.ndarray) -> dict:
    """
    Read header bits from the stego band (addition method).
    Compares stego vs original to detect ± perturbations.
    Returns the unpacked meta dict.
    """
    # Must pass BIT_OFFSET explicitly — header lives there, not at PAYLOAD_OFFSET
    hdr_bits  = decode_text_bits_addition(hf_stego, hf_original,
                                          HEADER_BITS, offset=BIT_OFFSET)
    hdr_bytes = np.packbits(hdr_bits.flatten())[:HEADER_BYTES]
    return _unpack_header(hdr_bytes)


def _decode_header_bits_override(hf_stego: np.ndarray) -> dict:
    """
    Read header bits from the stego band (override method).
    Uses sign of coefficients to detect bits — no original audio needed.
    """
    # Must pass BIT_OFFSET explicitly — header lives there, not at PAYLOAD_OFFSET
    hdr_bits  = decode_text_bits_override(hf_stego, HEADER_BITS, offset=BIT_OFFSET)
    hdr_bytes = np.packbits(hdr_bits.flatten())[:HEADER_BYTES]
    return _unpack_header(hdr_bytes)


# Payload bits start after the header in the DWT band
PAYLOAD_OFFSET = BIT_OFFSET + HEADER_BITS   # = 10 + 136 = 146


def text_bytes_to_bits(data: np.ndarray) -> np.ndarray:
    """Convert uint8 array to flat bit array (MSB first)."""
    return np.unpackbits(data)


def bits_to_text_bytes(bits: np.ndarray, n_bytes: int) -> np.ndarray:
    """Convert flat bit array back to uint8 array, take first n_bytes."""
    padded = np.zeros(math.ceil(len(bits) / 8) * 8, dtype=np.uint8)
    padded[:len(bits)] = bits
    return np.packbits(padded)[:n_bytes]


def embed_text_bits_addition(hf_band: np.ndarray,
                              bits: np.ndarray) -> np.ndarray:
    """
    Encode payload bits into the DWT band using ±BIT_SCALE perturbations.
    Starts at PAYLOAD_OFFSET (after the self-describing header region).
    bit=1 → +BIT_SCALE, bit=0 → -BIT_SCALE
    Decoding requires the original band to detect the sign of the change.
    """
    result = hf_band.copy()
    for i, bit in enumerate(bits):
        idx         = i + PAYLOAD_OFFSET
        result[idx] = hf_band[idx] + BIT_SCALE * (1 if bit else -1)
    return result


def decode_text_bits_addition(hf_stego: np.ndarray,
                               hf_original: np.ndarray,
                               n_bits: int,
                               offset: int = None) -> np.ndarray:
    """
    Recover n_bits from the stego band by comparing with the original band.
    offset defaults to PAYLOAD_OFFSET (after the header).
    Pass BIT_OFFSET explicitly to read the header instead.
    bit = 1 if stego_coeff > original_coeff, else 0.
    """
    if offset is None:
        offset = PAYLOAD_OFFSET
    diff = hf_stego[offset : offset + n_bits] \
         - hf_original[offset : offset + n_bits]
    return (diff > 0).astype(np.uint8)


def embed_text_bits_override(hf_band: np.ndarray,
                              bits: np.ndarray) -> tuple:
    """
    Encode payload bits into the DWT band WITHOUT needing the original audio.
    Sets each coefficient to +baseline (bit=1) or -baseline (bit=0).
    Starts at PAYLOAD_OFFSET (after the self-describing header region).
    Returns (modified band, baseline) — baseline is stored in the header.
    """
    result   = hf_band.copy()
    baseline = np.abs(hf_band).mean() + 0.1
    for i, bit in enumerate(bits):
        idx         = i + PAYLOAD_OFFSET
        result[idx] = baseline * (1 if bit else -1)
    return result, baseline


def decode_text_bits_override(hf_stego: np.ndarray,
                               n_bits: int,
                               offset: int = None) -> np.ndarray:
    """
    Recover n_bits from the stego band without original audio.
    offset defaults to PAYLOAD_OFFSET (after the header).
    Pass BIT_OFFSET explicitly to read the header instead.
    bit = 1 if coeff > 0, else 0.
    """
    if offset is None:
        offset = PAYLOAD_OFFSET
    region = hf_stego[offset : offset + n_bits]
    return (region > 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DWT DECOMPOSITION / RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def dwt_decompose(audio: np.ndarray,
                  wavelet: str = 'db4', level: int = 3) -> list:
    """
    Multi-level DWT. Returns [approx, detail_L, detail_L-1, ..., detail_1].
    coeffs[1] = finest detail = highest frequencies = most inaudible.
    """
    return pywt.wavedec(audio, wavelet=wavelet, level=level)


def dwt_reconstruct(coeffs: list, wavelet: str = 'db4') -> np.ndarray:
    """Reconstruct audio from DWT coefficients (IDWT)."""
    return pywt.waverec(coeffs, wavelet=wavelet)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EMBED  (addition method)
# ─────────────────────────────────────────────────────────────────────────────

def embed_addition(cover_audio: np.ndarray,
                   secret,
                   alpha: float       = 0.01,
                   wavelet: str       = 'db4',
                   dwt_level: int     = 3,
                   nperseg: int       = 256,
                   noverlap: int      = 128,
                   encrypt: bool      = True,
                   henon_params: dict = None,
                   arnold_iterations: int = 5) -> tuple:
    """
    ADDITION METHOD — embed a secret by ADDING to the high-frequency DWT band.

    Accepts either:
      secret = str          → text payload (exact lossless embedding)
      secret = np.ndarray   → grayscale image (approximate STFT embedding)

    Returns:
        (stego_audio)
        stego_audio : 1D float64 array  — the audio with hidden payload
        meta        : dict              — pass this to decode_addition()
    """
    if henon_params is None:
        henon_params = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}

    coeffs = dwt_decompose(cover_audio, wavelet=wavelet, level=dwt_level)
    hf_orig = coeffs[1].copy()

    if isinstance(secret, str):
        # ── TEXT PATH ──────────────────────────────────────────────────
        raw_bytes    = np.frombuffer(secret.encode("utf-8"), dtype=np.uint8).copy()
        payload_len  = len(raw_bytes)

        # Pack bytes into a square grid for chaotic encryption
        grid, _      = pad_to_square(raw_bytes)

        if encrypt:
            grid = encrypt_grid(grid, henon_params, arnold_iterations)

        # Embed ALL grid bytes (including zero-padding) so chaotic maps can
        # be fully reversed on decode — they operate on the complete square grid.
        all_bytes    = grid.flatten()
        bits         = text_bytes_to_bits(all_bytes)

        n_bits       = len(bits)
        needed       = n_bits + BIT_OFFSET
        assert needed <= len(coeffs[1]), (
            f"Text too long: needs {needed} HF coefficients, "
            f"audio only has {len(coeffs[1])}. Use longer audio.")

        # Build and embed the self-describing header first
        hdr_arr   = _pack_header(0, payload_len, grid.shape[0],
                                  0, 0, n_bits, alpha=alpha)
        hf_mod    = _embed_header_bits(hf_orig, hdr_arr)
        # Then embed the payload bits after the header
        coeffs[1] = embed_text_bits_addition(hf_mod, bits)

    else:
        # ── IMAGE PATH ─────────────────────────────────────────────────
        image        = secret
        orig_shape   = image.shape
        square, _    = image_to_square(image)

        if encrypt:
            square = encrypt_grid(square, henon_params, arnold_iterations)

        secret_sig   = grid_to_audio_signal(square, nperseg=nperseg, noverlap=noverlap)
        target_len   = len(coeffs[1])
        if len(secret_sig) >= target_len:
            trimmed  = secret_sig[:target_len]
        else:
            trimmed  = np.pad(secret_sig, (0, target_len - len(secret_sig)))

        coeffs[1]    = coeffs[1] + alpha * trimmed

        # Embed header into the first HEADER_BITS positions of the HF band
        hdr_arr      = _pack_header(1, orig_shape[0] * orig_shape[1],
                                     square.shape[0],
                                     orig_shape[0], orig_shape[1],
                                     0, alpha=alpha)
        coeffs[1]    = _embed_header_bits(coeffs[1], hdr_arr)

    stego_audio = dwt_reconstruct(coeffs, wavelet=wavelet)[:len(cover_audio)]
    return stego_audio


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DECODE  (addition method)
# ─────────────────────────────────────────────────────────────────────────────

def decode_addition(stego_audio: np.ndarray,
                    original_audio: np.ndarray,
                    wavelet: str       = 'db4',
                    dwt_level: int     = 3,
                    nperseg: int       = 256,
                    noverlap: int      = 128,
                    decrypt: bool      = True,
                    henon_params: dict = None,
                    arnold_iterations: int = 5):
    """
    ADDITION METHOD DECODING.
    Subtracts original audio from stego to isolate the hidden signal.
    Requires the original clean cover audio.

    No meta dict needed — all decoding information is read automatically
    from the self-describing header embedded in the audio by embed_addition().

    Returns:
        str          if payload was text
        np.ndarray   if payload was image
    """
    if henon_params is None:
        henon_params = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}

    coeffs_stego    = dwt_decompose(stego_audio,    wavelet=wavelet, level=dwt_level)
    coeffs_original = dwt_decompose(original_audio, wavelet=wavelet, level=dwt_level)
    hf_stego        = coeffs_stego[1]
    hf_original     = coeffs_original[1]

    # Read the self-describing header — no meta needed from the caller
    meta = _decode_header_bits_addition(hf_stego, hf_original)

    if meta["mode"] == "text":
        # ── TEXT PATH ──────────────────────────────────────────────────
        n_bits        = meta["n_bits"]
        payload_len   = meta["payload_len"]
        grid_shape    = meta["grid_shape"]

        bits          = decode_text_bits_addition(hf_stego, hf_original, n_bits)
        all_enc_bytes = bits_to_text_bytes(bits, grid_shape[0] * grid_shape[1])
        enc_grid      = all_enc_bytes.reshape(grid_shape)

        if decrypt:
            dec_grid  = decrypt_grid(enc_grid, henon_params, arnold_iterations)
        else:
            dec_grid  = enc_grid

        raw_bytes     = dec_grid.flatten()[:payload_len]
        return bytes(raw_bytes.tolist()).decode("utf-8", errors="replace")

    else:
        # ── IMAGE PATH ─────────────────────────────────────────────────
        alpha         = meta["alpha"]
        grid_shape    = meta["grid_shape"]
        H, W          = meta["orig_shape"]
        orig_shape    = (H, W)
        min_len       = min(len(hf_stego), len(hf_original))
        secret_sig    = (hf_stego[:min_len] - hf_original[:min_len]) / alpha

        enc_grid      = audio_signal_to_grid(secret_sig, grid_shape,
                                              nperseg=nperseg, noverlap=noverlap)
        if decrypt:
            dec_grid  = decrypt_grid(enc_grid, henon_params, arnold_iterations)
        else:
            dec_grid  = enc_grid

        H, W          = orig_shape
        return dec_grid[:H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EMBED  (override method)
# ─────────────────────────────────────────────────────────────────────────────

def embed_override(cover_audio: np.ndarray,
                   secret,
                   wavelet: str       = 'db4',
                   dwt_level: int     = 3,
                   nperseg: int       = 256,
                   noverlap: int      = 128,
                   encrypt: bool      = True,
                   henon_params: dict = None,
                   arnold_iterations: int = 5) -> tuple:
    """
    OVERRIDE METHOD — embed a secret by REPLACING the high-frequency DWT band.

    Returns:
        (stego_audio)
        stego_audio : 1D float64 array
        meta        : dict — pass this to decode_override()
    """
    if henon_params is None:
        henon_params = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}

    coeffs  = dwt_decompose(cover_audio, wavelet=wavelet, level=dwt_level)
    hf_orig = coeffs[1].copy()

    if isinstance(secret, str):
        # ── TEXT PATH ──────────────────────────────────────────────────
        raw_bytes    = np.frombuffer(secret.encode("utf-8"), dtype=np.uint8).copy()
        payload_len  = len(raw_bytes)
        grid, _      = pad_to_square(raw_bytes)

        if encrypt:
            grid = encrypt_grid(grid, henon_params, arnold_iterations)

        # Embed ALL grid bytes so the full grid can be reconstructed on decode
        all_bytes    = grid.flatten()
        bits         = text_bytes_to_bits(all_bytes)
        n_bits       = len(bits)

        new_hf, baseline = embed_text_bits_override(hf_orig, bits)
        # Embed header in the same override style (sign-based, no original needed)
        hdr_arr   = _pack_header(0, payload_len, grid.shape[0],
                                  0, 0, n_bits)
        hdr_bits  = np.unpackbits(hdr_arr)
        for i, bit in enumerate(hdr_bits):
            idx         = BIT_OFFSET + i
            new_hf[idx] = baseline * (1 if bit else -1)
        coeffs[1] = new_hf

    else:
        # ── IMAGE PATH ─────────────────────────────────────────────────
        image        = secret
        orig_shape   = image.shape
        square, _    = image_to_square(image)

        if encrypt:
            square = encrypt_grid(square, henon_params, arnold_iterations)

        secret_sig   = grid_to_audio_signal(square, nperseg=nperseg, noverlap=noverlap)
        target_len   = len(coeffs[1])
        if len(secret_sig) >= target_len:
            trimmed  = secret_sig[:target_len]
        else:
            trimmed  = np.pad(secret_sig, (0, target_len - len(secret_sig)))

        coeff_scale  = np.std(hf_orig) / (np.std(trimmed) + 1e-10)
        coeffs[1]    = trimmed * coeff_scale

        # Embed header at the start of the HF band
        hdr_arr      = _pack_header(1, orig_shape[0] * orig_shape[1],
                                     square.shape[0],
                                     orig_shape[0], orig_shape[1],
                                     0, coeff_scale=coeff_scale)
        coeffs[1]    = _embed_header_bits(coeffs[1], hdr_arr)

    stego_audio = dwt_reconstruct(coeffs, wavelet=wavelet)[:len(cover_audio)]
    return stego_audio


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — DECODE  (override method)
# ─────────────────────────────────────────────────────────────────────────────

def decode_override(stego_audio: np.ndarray,
                    wavelet: str       = 'db4',
                    dwt_level: int     = 3,
                    nperseg: int       = 256,
                    noverlap: int      = 128,
                    decrypt: bool      = True,
                    henon_params: dict = None,
                    arnold_iterations: int = 5):
    """
    OVERRIDE METHOD DECODING.
    Does NOT require the original audio.

    No meta dict needed — all decoding information is read automatically
    from the self-describing header embedded in the audio by embed_override().

    Returns:
        str          if payload was text
        np.ndarray   if payload was image
    """
    if henon_params is None:
        henon_params = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}

    coeffs = dwt_decompose(stego_audio, wavelet=wavelet, level=dwt_level)
    hf     = coeffs[1]

    # Read the self-describing header — no meta needed from the caller
    meta = _decode_header_bits_override(hf)

    if meta["mode"] == "text":
        # ── TEXT PATH ──────────────────────────────────────────────────
        n_bits        = meta["n_bits"]
        payload_len   = meta["payload_len"]
        grid_shape    = meta["grid_shape"]

        bits          = decode_text_bits_override(hf, n_bits)
        all_enc_bytes = bits_to_text_bytes(bits, grid_shape[0] * grid_shape[1])
        enc_grid      = all_enc_bytes.reshape(grid_shape)

        if decrypt:
            dec_grid = decrypt_grid(enc_grid, henon_params, arnold_iterations)
        else:
            dec_grid = enc_grid

        raw_bytes     = dec_grid.flatten()[:payload_len]
        return bytes(raw_bytes.tolist()).decode("utf-8", errors="replace")

    else:
        # ── IMAGE PATH ─────────────────────────────────────────────────
        coeff_scale = meta["coeff_scale"]
        grid_shape  = meta["grid_shape"]
        H, W        = meta["orig_shape"]
        orig_shape  = (H, W)

        secret_sig  = hf / coeff_scale
        enc_grid    = audio_signal_to_grid(secret_sig, grid_shape,
                                            nperseg=nperseg, noverlap=noverlap)

        if decrypt:
            dec_grid = decrypt_grid(enc_grid, henon_params, arnold_iterations)
        else:
            dec_grid = enc_grid

        return dec_grid[:H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, recovered: np.ndarray,
                 max_val: float = 255.0) -> float:
    """PSNR in dB. Higher = better image quality."""
    mse = np.mean((original.astype(np.float64) - recovered.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10((max_val ** 2) / mse)


def compute_mse(original: np.ndarray, recovered: np.ndarray) -> float:
    """Mean Squared Error. Lower = better."""
    return float(np.mean((original.astype(np.float64) - recovered.astype(np.float64)) ** 2))


def compute_audio_snr(original: np.ndarray, stego: np.ndarray) -> float:
    """Audio SNR in dB. Higher = audio quality better preserved."""
    sig  = np.sum(original ** 2)
    noise = np.sum((original - stego) ** 2)
    return float('inf') if noise == 0 else 10 * np.log10(sig / noise)


def text_match_score(original: str, recovered: str) -> float:
    """Percentage of characters that match. 100.0 = perfect recovery."""
    if not original:
        return 100.0
    matches = sum(a == b for a, b in zip(original, recovered[:len(original)]))
    return 100.0 * matches / len(original)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_image(N: int = 64) -> np.ndarray:
    """Synthetic N×N grayscale test image with gradient + circle."""
    img    = np.zeros((N, N), dtype=np.uint8)
    center = N // 2
    for i in range(N):
        for j in range(N):
            img[i, j] = int((i + j) / (2 * N) * 200)
            if abs((i - center) ** 2 + (j - center) ** 2 - (N // 3) ** 2) < 30:
                img[i, j] = 255
    return img


def _make_cover_audio(duration: float = 2.0,
                      sample_rate: int = 44100) -> np.ndarray:
    """Synthetic two-tone jingle with noise."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return (0.5 * np.sin(2 * np.pi * 440 * t) +
            0.3 * np.sin(2 * np.pi * 880 * t) +
            0.1 * np.random.randn(len(t))).astype(np.float64)


def run_demo():
    """
    Run all four combinations:
      1. Image  ×  Addition method
      2. Image  ×  Override method
      3. Text   ×  Addition method
      4. Text   ×  Override method
    """
    print("=" * 70)
    print("  AUDIO STEGANOGRAPHY — IMAGE & TEXT ENCRYPTION DEMO")
    print("  Based on Nasr et al. (2024), Scientific Reports 14:22054")
    print("=" * 70)

    henon_params      = {"x0": 0.01, "x1": 0.02, "a": 0.3, "b": 1.4}
    arnold_iterations = 5
    alpha             = 0.05

    print("\n  Secret key:")
    print(f"    Henon  : x0={henon_params['x0']}, x1={henon_params['x1']}, "
          f"a={henon_params['a']}, b={henon_params['b']}")
    print(f"    Arnold : {arnold_iterations} iterations")
    print(f"    Baker  : auto-generated from payload size")
    print(f"    Alpha  : {alpha}  (addition method strength)")

    img         = _make_test_image(64)
    secret_text = (
        "This is a secret message hidden inside an audio file using "
        "chaotic map encryption and DWT-based audio steganography. "
        "Based on Nasr et al. (2024)."
    )
    cover_audio = _make_cover_audio()

    print(f"\n  Secret image : {img.shape[0]}×{img.shape[1]} grayscale")
    print(f"  Secret text  : {len(secret_text)} chars — \"{secret_text[:55]}...\"")
    print(f"  Cover audio  : {len(cover_audio)} samples (2 s at 44.1 kHz)")

    results = {}

    # ── [1/4] Image + Addition ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [1/4]  IMAGE  ×  ADDITION  (requires original audio to decode)")
    print("─" * 70)

    stego = embed_addition(cover_audio, img, alpha=alpha,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)
    snr = compute_audio_snr(cover_audio, stego[:len(cover_audio)])
    print(f"  Embedded.   Audio SNR = {snr:.2f} dB")

    recovered = decode_addition(stego, cover_audio,
                                henon_params=henon_params,
                                arnold_iterations=arnold_iterations)
    psnr = compute_psnr(img, recovered)
    mse  = compute_mse(img, recovered)
    print(f"  Decoded.    Image PSNR = {psnr:.2f} dB   MSE = {mse:.2f}")
    print(f"  Shape check: {recovered.shape}  (expected {img.shape})")
    results["img_add"] = dict(snr=snr, psnr=psnr, mse=mse, needs_orig=True)

    # ── [2/4] Image + Override ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [2/4]  IMAGE  ×  OVERRIDE  (no original audio needed)")
    print("─" * 70)

    stego = embed_override(cover_audio, img,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)
    snr = compute_audio_snr(cover_audio, stego[:len(cover_audio)])
    print(f"  Embedded.   Audio SNR = {snr:.2f} dB")

    recovered = decode_override(stego,
                                henon_params=henon_params,
                                arnold_iterations=arnold_iterations)
    psnr = compute_psnr(img, recovered)
    mse  = compute_mse(img, recovered)
    print(f"  Decoded.    Image PSNR = {psnr:.2f} dB   MSE = {mse:.2f}")
    print(f"  Shape check: {recovered.shape}  (expected {img.shape})")
    results["img_ovr"] = dict(snr=snr, psnr=psnr, mse=mse, needs_orig=False)

    # ── [3/4] Text + Addition ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [3/4]  TEXT  ×  ADDITION  (requires original audio to decode)")
    print("─" * 70)

    stego = embed_addition(cover_audio, secret_text, alpha=alpha,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)
    snr = compute_audio_snr(cover_audio, stego[:len(cover_audio)])
    print(f"  Embedded.   Audio SNR = {snr:.2f} dB")

    recovered = decode_addition(stego, cover_audio,
                                henon_params=henon_params,
                                arnold_iterations=arnold_iterations)
    match = text_match_score(secret_text, recovered)
    print(f"  Decoded.    Character match = {match:.1f}%")
    print(f"  Recovered:  \"{recovered[:65]}...\"")
    results["txt_add"] = dict(snr=snr, match=match, needs_orig=True)

    # ── [4/4] Text + Override ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [4/4]  TEXT  ×  OVERRIDE  (no original audio needed)")
    print("─" * 70)

    stego = embed_override(cover_audio, secret_text,
                                 henon_params=henon_params,
                                 arnold_iterations=arnold_iterations)
    snr = compute_audio_snr(cover_audio, stego[:len(cover_audio)])
    print(f"  Embedded.   Audio SNR = {snr:.2f} dB")

    recovered = decode_override(stego,
                                henon_params=henon_params,
                                arnold_iterations=arnold_iterations)
    match = text_match_score(secret_text, recovered)
    print(f"  Decoded.    Character match = {match:.1f}%")
    print(f"  Recovered:  \"{recovered[:65]}...\"")
    results["txt_ovr"] = dict(snr=snr, match=match, needs_orig=False)

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    w = 35
    print(f"  {'Combination':<{w}} {'Audio SNR':>10} {'Quality':>14} {'Needs orig':>11}")
    print(f"  {'-'*w} {'-'*10} {'-'*14} {'-'*11}")

    r = results["img_add"]
    q = "PSNR " + str(round(r["psnr"], 1)) + " dB"
    print(f"  {'Image  + Addition':<{w}} {r['snr']:>9.2f}  {q:>14} {'YES':>11}")

    r = results["img_ovr"]
    q = "PSNR " + str(round(r["psnr"], 1)) + " dB"
    print(f"  {'Image  + Override':<{w}} {r['snr']:>9.2f}  {q:>14} {'NO':>11}")

    r = results["txt_add"]
    q = "Match " + str(round(r["match"])) + "%"
    print(f"  {'Text   + Addition':<{w}} {r['snr']:>9.2f}  {q:>14} {'YES':>11}")

    r = results["txt_ovr"]
    q = "Match " + str(round(r["match"])) + "%"
    print(f"  {'Text   + Override':<{w}} {r['snr']:>9.2f}  {q:>14} {'NO':>11}")

    print("=" * 70)

    # ── Save images ───────────────────────────────────────────────────
    PILImage.fromarray(img).save("/mnt/user-data/outputs/original_image.png")
    PILImage.fromarray(results["img_add"].get("recovered",
        decode_addition(stego, cover_audio,
                       henon_params=henon_params,
                       arnold_iterations=arnold_iterations)
        if False else img)).save("/mnt/user-data/outputs/placeholder.png")
    print("\n  Saved: original_image.png")
    print()


if __name__ == "__main__":
    run_demo()
