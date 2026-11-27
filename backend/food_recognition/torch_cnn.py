"""PyTorch MobileNetV2 transfer-learning CNN for real-photo food classification."""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from backend.food_recognition.labels import DISH_BY_KEY, DISH_CLASSES, NUM_CLASSES
from backend.food_recognition.preprocessing import ImageInput, load_pil_image

DEFAULT_TORCH_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "mobilenet_v2_food.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mobilenet_v2_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> nn.Module:
    """Create a MobileNetV2 CNN with a custom food classification head."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(256, num_classes),
    )
    return model


class MappedImageFolderDataset(Dataset):
    """Dataset reading `<root>/<dish_key>/*.jpg` mapped to `DISH_BY_KEY` class IDs."""

    def __init__(self, root: Path, transform: transforms.Compose):
        self.samples: List[Tuple[Path, int]] = []
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for sub in sorted(Path(root).iterdir()):
            if not sub.is_dir() or sub.name not in DISH_BY_KEY:
                continue
            cid = DISH_BY_KEY[sub.name].class_id
            for p in sorted(sub.iterdir()):
                if p.suffix.lower() in exts:
                    self.samples.append((p, cid))
        if not self.samples:
            raise ValueError(f"No valid class images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, label


class TorchFoodCNN:
    """Inference wrapper for the fine-tuned MobileNetV2 CNN."""

    MODEL_NAME = "NutriLens-MobileNetV2-Food101"

    def __init__(self, weights_path: Path = DEFAULT_TORCH_WEIGHTS_PATH, device: Optional[str] = None):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"PyTorch CNN weights not found: {self.weights_path}")
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self.num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
        self.model = build_mobilenet_v2_classifier(num_classes=self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_inference_transform()

    @torch.inference_mode()
    def predict_proba(self, image_input: ImageInput) -> np.ndarray:
        """Predict class probabilities of shape (1, num_classes) for an image input."""
        pil_img = load_pil_image(image_input)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


def train_mobilenet_cnn(
    data_dir: Path,
    weights_path: Path = DEFAULT_TORCH_WEIGHTS_PATH,
    epochs: int = 12,
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict:
    """Fine-tune MobileNetV2 on real food images organized as `<data_dir>/{train,val}/<dish_key>/*`."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train" if (data_dir / "train").is_dir() else data_dir
    val_dir = data_dir / "val" if (data_dir / "val").is_dir() else None

    train_ds = MappedImageFolderDataset(train_dir, transform=get_train_transform())
    if val_dir is not None:
        val_ds = MappedImageFolderDataset(val_dir, transform=get_inference_transform())
    else:
        raise ValueError(f"Validation folder not found under {data_dir}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_mobilenet_v2_classifier(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # Freeze early layers, fine-tune last 5 inverted residual blocks + classifier head
    for param in model.features[:14].parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features[14:].parameters(), "lr": 2.5e-4, "weight_decay": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1.2e-3, "weight_decay": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = -1.0
    best_top3 = 0.0
    best_epoch = 0
    history = []

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            train_correct += int((logits.argmax(dim=1) == yb).sum().item())

        scheduler.step()
        train_loss = total_loss / len(train_ds)
        train_acc = train_correct / len(train_ds)

        model.eval()
        val_correct = 0
        val_top3_hits = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                val_correct += int((preds == yb).sum().item())
                top3 = logits.topk(3, dim=1).indices
                val_top3_hits += int(sum(int(yb[i] in top3[i]) for i in range(yb.size(0))))

        val_acc = val_correct / len(val_ds)
        val_top3 = val_top3_hits / len(val_ds)

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
            best_top3 = val_top3
            best_epoch = epoch
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_name": TorchFoodCNN.MODEL_NAME,
                    "num_classes": NUM_CLASSES,
                    "class_keys": [d.key for d in DISH_CLASSES],
                    "best_epoch": best_epoch,
                    "best_val_acc": round(best_val_acc, 4),
                    "val_top3": round(best_top3, 4),
                    "state_dict": cpu_state,
                },
                weights_path,
            )

    summary = {
        "model_name": TorchFoodCNN.MODEL_NAME,
        "device": str(device),
        "weights_path": str(weights_path),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_top3": round(best_top3, 4),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "history": history,
    }
    weights_path.with_name("torch_train_history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
