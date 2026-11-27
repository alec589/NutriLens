"""Training script for NutriLens FoodCNN.

Supports:
1. Built-in procedural food dataset (24 dish classes) for reproducible offline training.
2. Custom image folder dataset (`--data-dir <root>/<dish_key>/*.jpg`) for real-photo fine-tuning.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from backend.food_recognition.cnn_model import AdamOptimizer, FoodCNN, _softmax
from backend.food_recognition.labels import DISH_BY_KEY, DISH_CLASSES, NUM_CLASSES
from backend.food_recognition.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    augment_pil_image,
    preprocess_image,
)

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "food_cnn.npz"


# Visual recipe descriptors (background/plate style + ingredient shapes & RGB palettes)
# used to synthesize diverse training & validation dish images for the lightweight NumPy CNN fallback.
_DISH_VISUAL_SPECS: Dict[str, Dict] = {
    "caesar_salad": {
        "vessel": "bowl",
        "base": (96, 168, 72),
        "accents": [((218, 188, 124), "squares", 8), ((244, 240, 224), "strips", 6), ((216, 172, 118), "strips", 5)],
    },
    "greek_salad": {
        "vessel": "bowl",
        "base": (88, 158, 66),
        "accents": [((246, 244, 238), "squares", 8), ((216, 52, 42), "chunks", 6), ((38, 34, 42), "berries", 7)],
    },
    "caprese_salad": {
        "vessel": "plate",
        "base": (244, 242, 236),
        "accents": [((216, 46, 38), "blobs", 5), ((248, 246, 236), "blobs", 5), ((48, 136, 52), "leaves", 6)],
    },
    "club_sandwich": {
        "vessel": "plate",
        "base": (214, 168, 102),
        "accents": [((78, 158, 62), "layers", 3), ((214, 52, 42), "layers", 2), ((242, 212, 162), "layers", 3)],
    },
    "grilled_cheese_sandwich": {
        "vessel": "plate",
        "base": (212, 148, 58),
        "accents": [((244, 198, 46), "strips", 5), ((154, 92, 34), "grill_lines", 5)],
    },
    "pulled_pork_sandwich": {
        "vessel": "plate",
        "base": (196, 134, 68),
        "accents": [((128, 48, 26), "blobs", 5), ((234, 224, 184), "specks", 12)],
    },
    "hamburger": {
        "vessel": "plate",
        "base": (208, 146, 68),
        "accents": [((92, 52, 34), "layers", 3), ((242, 192, 42), "layers", 2), ((76, 154, 58), "layers", 2)],
    },
    "hot_dog": {
        "vessel": "plate",
        "base": (228, 192, 128),
        "accents": [((176, 68, 44), "strips", 4), ((244, 206, 42), "stripes", 4), ((204, 38, 32), "stripes", 3)],
    },
    "pizza": {
        "vessel": "dark_plate",
        "base": (238, 212, 132),
        "accents": [((222, 178, 102), "ring", 1), ((184, 38, 32), "blobs", 9), ((208, 56, 38), "specks", 8)],
    },
    "macaroni_and_cheese": {
        "vessel": "bowl",
        "base": (244, 184, 44),
        "accents": [((218, 144, 28), "crescents", 14), ((184, 112, 24), "specks", 10)],
    },
    "spaghetti_bolognese": {
        "vessel": "plate",
        "base": (236, 208, 136),
        "accents": [((186, 48, 32), "blobs", 6), ((126, 62, 38), "specks", 14), ((244, 238, 216), "stripes", 6)],
    },
    "lasagna": {
        "vessel": "plate",
        "base": (204, 68, 38),
        "accents": [((242, 214, 134), "layers", 4), ((248, 242, 216), "blobs", 5), ((64, 144, 56), "specks", 6)],
    },
    "grilled_salmon": {
        "vessel": "plate",
        "base": (235, 128, 92),
        "accents": [((138, 72, 42), "grill_lines", 5), ((66, 152, 72), "clusters", 5), ((242, 222, 62), "blobs", 2)],
    },
    "steak": {
        "vessel": "dark_plate",
        "base": (106, 58, 38),
        "accents": [((74, 38, 24), "grill_lines", 6), ((58, 145, 62), "clusters", 5), ((232, 124, 38), "chunks", 4)],
    },
    "tacos": {
        "vessel": "plate",
        "base": (232, 188, 96),
        "accents": [((114, 68, 42), "crescents", 4), ((76, 158, 58), "specks", 12), ((214, 48, 38), "specks", 10)],
    },
    "breakfast_burrito": {
        "vessel": "plate",
        "base": (236, 214, 176),
        "accents": [((244, 198, 56), "chunks", 7), ((46, 42, 46), "specks", 9), ((224, 146, 42), "strips", 5)],
    },
    "pancakes": {
        "vessel": "plate",
        "base": (212, 154, 82),
        "accents": [((162, 88, 32), "blobs", 4), ((56, 48, 118), "berries", 9), ((244, 218, 96), "squares", 2)],
    },
    "french_toast": {
        "vessel": "plate",
        "base": (198, 134, 62),
        "accents": [((246, 244, 238), "specks", 12), ((188, 42, 54), "berries", 6), ((148, 74, 26), "blobs", 3)],
    },
    "eggs_benedict": {
        "vessel": "plate",
        "base": (244, 206, 68),
        "accents": [((248, 244, 234), "blobs", 4), ((214, 124, 118), "strips", 4), ((56, 142, 58), "specks", 6)],
    },
    "omelette": {
        "vessel": "plate",
        "base": (244, 202, 58),
        "accents": [((218, 146, 44), "crescents", 4), ((64, 146, 56), "specks", 8), ((212, 56, 42), "specks", 6)],
    },
    "chicken_wings": {
        "vessel": "dark_plate",
        "base": (182, 62, 28),
        "accents": [((142, 42, 22), "blobs", 8), ((88, 168, 68), "strips", 4)],
    },
    "fried_rice": {
        "vessel": "bowl",
        "base": (235, 205, 128),
        "accents": [((245, 224, 92), "specks", 16), ((58, 150, 62), "specks", 12), ((220, 102, 42), "specks", 8)],
    },
    "ramen": {
        "vessel": "dark_bowl",
        "base": (216, 184, 128),
        "accents": [((238, 218, 158), "stripes", 9), ((248, 244, 232), "yolks", 2), ((162, 108, 72), "blobs", 3)],
    },
    "frozen_yogurt": {
        "vessel": "bowl",
        "base": (248, 247, 244),
        "accents": [((212, 52, 64), "berries", 7), ((68, 56, 128), "berries", 7), ((210, 168, 98), "specks", 10)],
    },
}


def _jitter_color(rgb: Tuple[int, int, int], rng: np.random.Generator, delta: int = 18) -> Tuple[int, int, int]:
    return tuple(int(np.clip(c + rng.integers(-delta, delta + 1), 0, 255)) for c in rgb)


def generate_synthetic_dish_image(
    dish_key: str,
    rng: np.random.Generator,
    size: int = 64,
) -> Image.Image:
    """Render a synthetic 64x64 food photo with vessel geometry, textures, and lighting variation."""
    spec = _DISH_VISUAL_SPECS[dish_key]
    table_palettes = [(164, 126, 88), (215, 208, 198), (74, 78, 84), (190, 165, 132)]
    bg = _jitter_color(table_palettes[int(rng.integers(0, len(table_palettes)))], rng, 14)
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    margin = int(rng.integers(5, 9))
    vessel = spec["vessel"]
    if vessel == "dark_bowl":
        rim_color = (42, 42, 48)
    elif vessel == "dark_plate":
        rim_color = (56, 60, 66)
    else:
        rim_color = (242, 242, 244)

    draw.ellipse([margin - 2, margin - 2, size - margin + 2, size - margin + 2], fill=_jitter_color(rim_color, rng, 10))
    inner = margin + 3
    base_rgb = _jitter_color(spec["base"], rng, 16)
    draw.ellipse([inner, inner, size - inner, size - inner], fill=base_rgb)

    cx, cy = size / 2.0, size / 2.0
    max_r = (size - 2 * inner) * 0.42

    for color, shape, count in spec["accents"]:
        if shape == "ring":
            c = _jitter_color(color, rng, 14)
            draw.ellipse([inner, inner, size - inner, size - inner], outline=c, width=4)
            continue
        if shape == "half_plate":
            c = _jitter_color(color, rng, 10)
            draw.pieslice([inner, inner, size - inner, size - inner], start=90, end=270, fill=c)
            continue

        for _ in range(count):
            angle = float(rng.uniform(0, 2 * np.pi))
            rad = float(rng.uniform(0, max_r))
            px = cx + rad * np.cos(angle)
            py = cy + rad * np.sin(angle)
            c = _jitter_color(color, rng, 16)

            if shape in ("squares", "chunks"):
                r = int(rng.integers(3, 6))
                draw.rectangle([px - r, py - r, px + r, py + r], fill=c)
            elif shape in ("clusters", "blobs", "berries"):
                r = int(rng.integers(2, 6)) if shape != "blobs" else int(rng.integers(4, 8))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=c)
            elif shape == "yolks":
                r = int(rng.integers(4, 6))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=c)
            elif shape == "strips":
                w = int(rng.integers(5, 9))
                h = int(rng.integers(2, 4))
                draw.rectangle([px - w, py - h, px + w, py + h], fill=c)
            elif shape == "stripes":
                y0 = int(cy + rng.uniform(-max_r, max_r))
                draw.line([inner + 3, y0, size - inner - 3, y0 + int(rng.integers(-4, 5))], fill=c, width=2)
            elif shape == "grill_lines":
                offset = int(rng.integers(-10, 11))
                draw.line([inner + 4, cy + offset - 5, size - inner - 4, cy + offset + 5], fill=c, width=2)
            elif shape == "crescents":
                r = int(rng.integers(4, 6))
                draw.chord([px - r, py - r, px + r, py + r], start=15, end=195, fill=c)
            elif shape == "layers":
                w = int(rng.integers(10, 15))
                h = int(rng.integers(2, 4))
                draw.rectangle([cx - w, py - h, cx + w, py + h], fill=c)
            elif shape in ("specks", "leaves"):
                r = int(rng.integers(1, 3)) if shape == "specks" else int(rng.integers(3, 5))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=c)

    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.8))))
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, 5.5, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_synthetic_dataset(
    samples_per_class: int = 48,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (X, y) tensors of shape (N, 3, 64, 64) and (N,)."""
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for dish in DISH_CLASSES:
        for _ in range(samples_per_class):
            img = generate_synthetic_dish_image(dish.key, rng, size=DEFAULT_IMAGE_SIZE[0])
            if rng.random() < 0.7:
                img = augment_pil_image(img, rng)
            xs.append(preprocess_image(img))
            ys.append(dish.class_id)
    idx = rng.permutation(len(ys))
    X = np.stack(xs, axis=0)[idx]
    y = np.array(ys, dtype=np.int64)[idx]
    return X, y


def load_image_folder_dataset(data_dir: Path, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load real images from `<data_dir>/<dish_key>/*` matching `DISH_CLASSES`."""
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or sub.name not in DISH_BY_KEY:
            continue
        cid = DISH_BY_KEY[sub.name].class_id
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in exts:
                xs.append(preprocess_image(p))
                ys.append(cid)
    if not xs:
        raise ValueError(f"No valid class folders/images found in {data_dir}")
    idx = rng.permutation(len(ys))
    return np.stack(xs, axis=0)[idx], np.array(ys, dtype=np.int64)[idx]


def cross_entropy_loss_and_grad(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute mean cross-entropy loss and gradient wrt logits."""
    probs = _softmax(logits)
    n = logits.shape[0]
    clipped = np.clip(probs[np.arange(n), targets], 1e-12, 1.0)
    loss = float(-np.mean(np.log(clipped)))
    dlogits = probs.copy()
    dlogits[np.arange(n), targets] -= 1.0
    dlogits /= n
    return loss, dlogits.astype(np.float32)


def train_food_cnn(
    epochs: int = 22,
    batch_size: int = 64,
    lr: float = 4e-3,
    samples_per_class: int = 50,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    data_dir: Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Train FoodCNN, validate on hold-out split, and save weights to disk."""
    if data_dir is not None:
        X, y = load_image_folder_dataset(data_dir, seed=seed)
    else:
        X, y = build_synthetic_dataset(samples_per_class=samples_per_class, seed=seed)

    n_total = len(y)
    n_val = max(NUM_CLASSES, int(0.2 * n_total))
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    model = FoodCNN(num_classes=NUM_CLASSES, dropout_rate=0.12, seed=seed)
    optimizer = AdamOptimizer(model, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    history = []
    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Cosine learning rate schedule
        optimizer.lr = lr * (0.2 + 0.8 * 0.5 * (1.0 + np.cos(np.pi * (epoch - 1) / max(epochs, 1))))
        perm = rng.permutation(len(y_train))
        epoch_losses = []
        correct = 0

        for start in range(0, len(y_train), batch_size):
            b_idx = perm[start : start + batch_size]
            xb, yb = X_train[b_idx], y_train[b_idx]
            logits = model.forward(xb, training=True)
            loss, dlogits = cross_entropy_loss_and_grad(logits, yb)
            model.backward(dlogits)
            optimizer.step()
            epoch_losses.append(loss)
            correct += int(np.sum(np.argmax(logits, axis=1) == yb))

        train_acc = correct / len(y_train)
        train_loss = float(np.mean(epoch_losses))

        val_probs = model.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = float(np.mean(val_preds == y_val))
        top3_idx = np.argsort(val_probs, axis=1)[:, -3:]
        val_top3 = float(np.mean([y_val[i] in top3_idx[i] for i in range(len(y_val))]))

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "val_top3": round(val_top3, 4),
            }
        )
        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | loss={train_loss:.4f} | "
                f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | val_top3={val_top3:.3f}"
            )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save_weights(
                weights_path,
                metadata={
                    "model_name": FoodCNN.MODEL_NAME,
                    "best_epoch": epoch,
                    "best_val_acc": f"{val_acc:.4f}",
                    "val_top3": f"{val_top3:.4f}",
                },
            )

    summary = {
        "weights_path": str(weights_path),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "history": history,
    }
    history_path = weights_path.with_name("train_history.json")
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NutriLens CNN model")
    parser.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--weights-path", type=Path, default=None)
    args = parser.parse_args()
    if args.backend == "torch":
        if args.data_dir is None:
            raise SystemExit("--data-dir is required when --backend torch")
        from backend.food_recognition.torch_cnn import DEFAULT_TORCH_WEIGHTS_PATH, train_mobilenet_cnn

        summary = train_mobilenet_cnn(
            data_dir=args.data_dir,
            weights_path=args.weights_path or DEFAULT_TORCH_WEIGHTS_PATH,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(
            f"Saved fine-tuned MobileNetV2 weights to {summary['weights_path']} "
            f"(best_val_acc={summary['best_val_acc']}, top3={summary['best_val_top3']})"
        )
    else:
        summary = train_food_cnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            samples_per_class=args.samples_per_class,
            weights_path=args.weights_path or DEFAULT_WEIGHTS_PATH,
            data_dir=args.data_dir,
        )
        print(f"Saved trained NumPy CNN weights to {summary['weights_path']} (best_val_acc={summary['best_val_acc']})")


if __name__ == "__main__":
    main()
