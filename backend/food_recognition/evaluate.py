"""Evaluation utilities and CLI for NutriLens CNN models (MobileNetV2 & NumPy FoodCNN)."""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score

from backend.food_recognition.cnn_model import FoodCNN
from backend.food_recognition.labels import DISH_BY_KEY, DISH_CLASSES, NUM_CLASSES
from backend.food_recognition.train_cnn import (
    DEFAULT_WEIGHTS_PATH,
    build_synthetic_dataset,
    load_image_folder_dataset,
)


def evaluate_torch_model(data_dir: Path, weights_path: Optional[Path] = None) -> Dict:
    """Evaluate the fine-tuned PyTorch MobileNetV2 CNN on real food images."""
    from backend.food_recognition.torch_cnn import DEFAULT_TORCH_WEIGHTS_PATH, TorchFoodCNN

    w_path = Path(weights_path or DEFAULT_TORCH_WEIGHTS_PATH)
    model = TorchFoodCNN(weights_path=w_path)

    val_root = Path(data_dir) / "val" if (Path(data_dir) / "val").is_dir() else Path(data_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples: List[tuple[Path, int]] = []
    for sub in sorted(val_root.iterdir()):
        if not sub.is_dir() or sub.name not in DISH_BY_KEY:
            continue
        cid = DISH_BY_KEY[sub.name].class_id
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in exts:
                samples.append((p, cid))
    if not samples:
        raise ValueError(f"No evaluation images found in {val_root}")

    y_true: List[int] = []
    probs_list: List[np.ndarray] = []
    t0 = time.perf_counter()
    for p, cid in samples:
        probs_list.append(model.predict_proba(p)[0])
        y_true.append(cid)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    probs = np.stack(probs_list, axis=0)
    y_test = np.array(y_true, dtype=np.int64)
    y_pred = np.argmax(probs, axis=1)

    top1_acc = float(np.mean(y_pred == y_test))
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = float(np.mean([y_test[i] in top3_idx[i] for i in range(len(y_test))]))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    per_class = {}
    for dish in DISH_CLASSES:
        mask = y_test == dish.class_id
        if np.any(mask):
            per_class[dish.key] = round(float(np.mean(y_pred[mask] == y_test[mask])), 4)

    metrics = {
        "model_name": TorchFoodCNN.MODEL_NAME,
        "backend": "torch_mobilenet_v2",
        "num_classes": NUM_CLASSES,
        "num_samples": int(len(y_test)),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "total_inference_ms": round(elapsed_ms, 2),
        "avg_latency_ms_per_image": round(elapsed_ms / max(len(y_test), 1), 3),
        "per_class_accuracy": per_class,
    }
    w_path.with_name("eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_model(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    data_dir: Optional[Path] = None,
    samples_per_class: int = 12,
    seed: int = 2026,
) -> Dict:
    """Evaluate NumPy FoodCNN on held-out synthetic or image-folder dataset."""
    model = FoodCNN.load_weights(weights_path)
    if data_dir is not None:
        X_test, y_test = load_image_folder_dataset(data_dir, seed=seed)
    else:
        X_test, y_test = build_synthetic_dataset(samples_per_class=samples_per_class, seed=seed)

    t0 = time.perf_counter()
    probs = model.predict_proba(X_test)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    y_pred = np.argmax(probs, axis=1)
    top1_acc = float(np.mean(y_pred == y_test))

    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = float(np.mean([y_test[i] in top3_idx[i] for i in range(len(y_test))]))

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    per_class = {}
    for dish in DISH_CLASSES:
        mask = y_test == dish.class_id
        if np.any(mask):
            per_class[dish.key] = round(float(np.mean(y_pred[mask] == y_test[mask])), 4)

    metrics = {
        "model_name": FoodCNN.MODEL_NAME,
        "backend": "numpy_cnn",
        "num_classes": NUM_CLASSES,
        "num_samples": int(len(y_test)),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "total_inference_ms": round(elapsed_ms, 2),
        "avg_latency_ms_per_image": round(elapsed_ms / max(len(y_test), 1), 3),
        "per_class_accuracy": per_class,
    }
    out_path = weights_path.with_name("eval_metrics.json")
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NutriLens CNN model")
    parser.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=12)
    args = parser.parse_args()
    if args.backend == "torch":
        if args.data_dir is None:
            raise SystemExit("--data-dir is required when --backend torch")
        metrics = evaluate_torch_model(data_dir=args.data_dir, weights_path=args.weights_path)
    else:
        metrics = evaluate_model(
            weights_path=args.weights_path or DEFAULT_WEIGHTS_PATH,
            data_dir=args.data_dir,
            samples_per_class=args.samples_per_class,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
