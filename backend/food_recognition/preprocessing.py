"""Image preprocessing and augmentation utilities for NutriLens CNN food recognition."""

import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ImageInput = Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (64, 64)

# ImageNet-like channel normalization constants in [0, 1] scale
CHANNEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
CHANNEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_pil_image(image_input: ImageInput) -> Image.Image:
    """Load an RGB PIL Image from path, raw bytes, file-like stream, PIL Image, or ndarray."""
    if isinstance(image_input, Image.Image):
        img = ImageOps.exif_transpose(image_input)
        return img.convert("RGB")
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    if isinstance(image_input, bytes):
        with Image.open(io.BytesIO(image_input)) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        with Image.open(image_input) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image_input type: {type(image_input)!r}")


def center_crop_square(image: Image.Image) -> Image.Image:
    """Center-crop an image to a square focusing on the plate/dish region."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def preprocess_image(
    image_input: ImageInput,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """Convert an input image to a float32 tensor of shape (3, H, W)."""
    img = load_pil_image(image_input)
    img = center_crop_square(img)
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    chw = np.transpose(arr, (2, 0, 1))  # (3, H, W)
    if normalize:
        chw = (chw - CHANNEL_MEAN) / CHANNEL_STD
    return np.ascontiguousarray(chw, dtype=np.float32)


def preprocess_batch(
    images: list,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess a list of images into a batch tensor of shape (N, 3, H, W)."""
    tensors = [preprocess_image(im, target_size=target_size, normalize=normalize) for im in images]
    return np.stack(tensors, axis=0)


def augment_pil_image(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply lightweight photometric and geometric augmentations for CNN training."""
    img = image.copy()
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.5:
        angle = float(rng.uniform(-18.0, 18.0))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)
    if rng.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.85, 1.20)))
    return img
