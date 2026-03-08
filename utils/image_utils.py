"""
utils/image_utils.py
---------------------
Shared image-processing helpers used across the image evaluation pipeline.

Public API
----------
  load_image_rgb(image_bytes)        – open bytes → (pil_img, rgb_f32, gray_f32)
  jpeg_recompress(img, quality)      – re-save as JPEG and decode → rgb_f32 array
  coeff_of_variation(arr, eps=1e-6)  – std / (mean + eps)
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


def load_image_rgb(
    image_bytes: bytes,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Open *image_bytes* as a PIL Image, convert to RGB, and return NumPy views.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes (JPEG, PNG, WEBP, TIFF, BMP, GIF …).

    Returns
    -------
    (pil_img, img_rgb, img_gray)
        pil_img  – PIL ``Image`` in RGB mode
        img_rgb  – ``float32`` ndarray shape (H, W, 3)
        img_gray – ``float32`` ndarray shape (H, W) — channel mean

    Raises
    ------
    Exception
        Any exception raised by PIL is propagated.  Callers should wrap
        this in a ``try / except`` and return a safe default result.
    """
    pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb  = np.array(pil_img, dtype=np.float32)
    img_gray = np.mean(img_rgb, axis=2)
    return pil_img, img_rgb, img_gray


def jpeg_recompress(img: Image.Image, quality: int) -> np.ndarray:
    """
    Re-compress *img* as JPEG at *quality* and return the result as a
    float32 RGB array.

    Parameters
    ----------
    img     : PIL.Image.Image  (RGB mode expected)
    quality : int  JPEG quality, 0–95

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype float32
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.float32)


def coeff_of_variation(arr: np.ndarray, eps: float = 1e-6) -> float:
    """
    Coefficient of variation: ``std(arr) / (mean(arr) + eps)``.

    Parameters
    ----------
    arr : np.ndarray  – any shape, numeric dtype
    eps : float       – small constant to prevent division by zero

    Returns
    -------
    float
    """
    return float(np.std(arr) / (float(np.mean(arr)) + eps))
