"""Pure-NumPy Convolutional Neural Network (FoodCNN) for NutriLens dish recognition.

Architecture (input shape: N x 3 x 64 x 64):
    Conv2D(3 -> 16, 3x3, pad=1) -> BatchNorm2D -> ReLU -> MaxPool2D(2)   [32x32x16]
    Conv2D(16 -> 32, 3x3, pad=1) -> BatchNorm2D -> ReLU -> MaxPool2D(2)  [16x16x32]
    Conv2D(32 -> 48, 3x3, pad=1) -> BatchNorm2D -> ReLU -> MaxPool2D(2)  [8x8x48]
    GlobalPool (GlobalAvgPool2D || GlobalMaxPool2D)                     [96]
    Dense(96 -> 64) -> ReLU -> Dropout -> Dense(64 -> num_classes) -> Softmax
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _relu_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = x > 0
    return np.where(mask, x, 0.0), mask


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class Conv2D:
    """2D Convolution layer with 3x3 kernel, padding=1, stride=1."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, seed: int = 42):
        if kernel_size != 3:
            raise ValueError("Conv2D currently optimizes kernel_size=3 with padding=1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.W = (rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * scale).astype(
            np.float32
        )
        self.b = np.zeros(out_channels, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache: Optional[Tuple[np.ndarray, int, int, int]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        cols = np.empty((n, c, 3, 3, h, w), dtype=np.float32)
        for kh in range(3):
            for kw in range(3):
                cols[:, :, kh, kw, :, :] = x_pad[:, :, kh : kh + h, kw : kw + w]
        # (N, H, W, C*9) -> (N*H*W, C*9)
        cols_2d = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * h * w, c * 9)
        w_2d = self.W.reshape(self.out_channels, c * 9).T  # (C*9, Out)
        out_2d = cols_2d @ w_2d + self.b  # (N*H*W, Out)
        out = out_2d.reshape(n, h, w, self.out_channels).transpose(0, 3, 1, 2)
        self._cache = (cols_2d, n, h, w)
        return np.ascontiguousarray(out, dtype=np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        cols_2d, n, h, w = self._cache
        dout_2d = dout.transpose(0, 2, 3, 1).reshape(n * h * w, self.out_channels)
        self.db = dout_2d.sum(axis=0).astype(np.float32)
        dw_2d = dout_2d.T @ cols_2d  # (Out, C*9)
        self.dW = dw_2d.reshape(self.W.shape).astype(np.float32)

        w_2d = self.W.reshape(self.out_channels, self.in_channels * 9)
        dcols_2d = dout_2d @ w_2d  # (N*H*W, C*9)
        dcols = dcols_2d.reshape(n, h, w, self.in_channels, 3, 3).transpose(0, 3, 4, 5, 1, 2)

        dx_pad = np.zeros((n, self.in_channels, h + 2, w + 2), dtype=np.float32)
        for kh in range(3):
            for kw in range(3):
                dx_pad[:, :, kh : kh + h, kw : kw + w] += dcols[:, :, kh, kw, :, :]
        return dx_pad[:, :, 1:-1, 1:-1]


class BatchNorm2D:
    """Spatial Batch Normalization over (N, H, W) per channel."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((1, num_features, 1, 1), dtype=np.float32)
        self.beta = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = (x - mean) * inv_std
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
            self._cache = (x_hat, inv_std, x - mean)
        else:
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
            x_hat = (x - self.running_mean) * inv_std
        return (self.gamma * x_hat + self.beta).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        x_hat, inv_std, _ = self._cache
        m = dout.shape[0] * dout.shape[2] * dout.shape[3]
        self.dbeta = dout.sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        self.dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True).astype(np.float32)

        dx_hat = dout * self.gamma
        dx = (1.0 / m) * inv_std * (
            m * dx_hat
            - dx_hat.sum(axis=(0, 2, 3), keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=(0, 2, 3), keepdims=True)
        )
        return dx.astype(np.float32)


class MaxPool2D:
    """2x2 Max Pooling with stride 2."""

    def __init__(self):
        self._cache: Optional[Tuple[Tuple[int, int, int, int], np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        x_reshaped = x.reshape(n, c, h // 2, 2, w // 2, 2)
        out = x_reshaped.max(axis=(3, 5))
        mask = x_reshaped == out[:, :, :, None, :, None]
        mask_norm = mask / np.maximum(mask.sum(axis=(3, 5), keepdims=True), 1)
        self._cache = (x.shape, mask_norm.astype(np.float32))
        return out.astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        orig_shape, mask_norm = self._cache
        dx_reshaped = mask_norm * dout[:, :, :, None, :, None]
        return dx_reshaped.reshape(orig_shape).astype(np.float32)


class GlobalDualPool2D:
    """Concatenates GlobalAveragePooling2D and GlobalMaxPooling2D across (H, W)."""

    def __init__(self):
        self._cache: Optional[Tuple[Tuple[int, int, int, int], np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        avg_pool = x.mean(axis=(2, 3))
        max_pool = x.max(axis=(2, 3))
        mask = x == max_pool[:, :, None, None]
        mask_norm = (mask / np.maximum(mask.sum(axis=(2, 3), keepdims=True), 1)).astype(np.float32)
        self._cache = (x.shape, mask_norm)
        return np.concatenate([avg_pool, max_pool], axis=1).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        orig_shape, mask_norm = self._cache
        n, c, h, w = orig_shape
        davg = dout[:, :c]
        dmax = dout[:, c:]
        dx_avg = np.broadcast_to((davg / (h * w))[:, :, None, None], orig_shape)
        dx_max = mask_norm * dmax[:, :, None, None]
        return (dx_avg + dx_max).astype(np.float32)


class Dense:
    """Fully connected linear layer."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_features)
        self.W = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return (x @ self.W + self.b).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self.dW = (self._x.T @ dout).astype(np.float32)
        self.db = dout.sum(axis=0).astype(np.float32)
        return (dout @ self.W.T).astype(np.float32)


class FoodCNN:
    """Multi-layer Convolutional Neural Network for NutriLens dish classification."""

    MODEL_NAME = "NutriLens-FoodCNN-v1"

    def __init__(self, num_classes: int = 24, dropout_rate: float = 0.15, seed: int = 42):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)

        self.conv1 = Conv2D(3, 16, seed=seed + 1)
        self.bn1 = BatchNorm2D(16)
        self.pool1 = MaxPool2D()

        self.conv2 = Conv2D(16, 32, seed=seed + 2)
        self.bn2 = BatchNorm2D(32)
        self.pool2 = MaxPool2D()

        self.conv3 = Conv2D(32, 48, seed=seed + 3)
        self.bn3 = BatchNorm2D(48)
        self.pool3 = MaxPool2D()

        self.global_pool = GlobalDualPool2D()  # 48 * 2 = 96 features
        self.fc1 = Dense(96, 64, seed=seed + 4)
        self.fc2 = Dense(64, num_classes, seed=seed + 5)

        self._relu_masks: List[np.ndarray] = []
        self._dropout_mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Run forward pass and return raw class logits of shape (N, num_classes)."""
        h = self.conv1.forward(x)
        h = self.bn1.forward(h, training=training)
        h, m1 = _relu_forward(h)
        h = self.pool1.forward(h)

        h = self.conv2.forward(h)
        h = self.bn2.forward(h, training=training)
        h, m2 = _relu_forward(h)
        h = self.pool2.forward(h)

        h = self.conv3.forward(h)
        h = self.bn3.forward(h, training=training)
        h, m3 = _relu_forward(h)
        h = self.pool3.forward(h)

        h = self.global_pool.forward(h)
        h = self.fc1.forward(h)
        h, m4 = _relu_forward(h)

        if training and self.dropout_rate > 0.0:
            keep = 1.0 - self.dropout_rate
            drop_mask = (self.rng.random(h.shape) < keep).astype(np.float32) / keep
            h = h * drop_mask
            self._dropout_mask = drop_mask
        else:
            self._dropout_mask = None

        logits = self.fc2.forward(h)
        self._relu_masks = [m1, m2, m3, m4]
        return logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probability distribution of shape (N, num_classes)."""
        logits = self.forward(x, training=False)
        return _softmax(logits)

    def backward(self, dlogits: np.ndarray) -> None:
        """Backpropagate gradient from logits through all layers."""
        m1, m2, m3, m4 = self._relu_masks
        dh = self.fc2.backward(dlogits)
        if self._dropout_mask is not None:
            dh = dh * self._dropout_mask
        dh = dh * m4
        dh = self.fc1.backward(dh)
        dh = self.global_pool.backward(dh)

        dh = self.pool3.backward(dh)
        dh = dh * m3
        dh = self.bn3.backward(dh)
        dh = self.conv3.backward(dh)

        dh = self.pool2.backward(dh)
        dh = dh * m2
        dh = self.bn2.backward(dh)
        dh = self.conv2.backward(dh)

        dh = self.pool1.backward(dh)
        dh = dh * m1
        dh = self.bn1.backward(dh)
        self.conv1.backward(dh)

    def param_groups(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Return trainable parameter tensors and their gradient buffers."""
        return [
            ("conv1_W", self.conv1.W, self.conv1.dW),
            ("conv1_b", self.conv1.b, self.conv1.db),
            ("bn1_gamma", self.bn1.gamma, self.bn1.dgamma),
            ("bn1_beta", self.bn1.beta, self.bn1.dbeta),
            ("conv2_W", self.conv2.W, self.conv2.dW),
            ("conv2_b", self.conv2.b, self.conv2.db),
            ("bn2_gamma", self.bn2.gamma, self.bn2.dgamma),
            ("bn2_beta", self.bn2.beta, self.bn2.dbeta),
            ("conv3_W", self.conv3.W, self.conv3.dW),
            ("conv3_b", self.conv3.b, self.conv3.db),
            ("bn3_gamma", self.bn3.gamma, self.bn3.dgamma),
            ("bn3_beta", self.bn3.beta, self.bn3.dbeta),
            ("fc1_W", self.fc1.W, self.fc1.dW),
            ("fc1_b", self.fc1.b, self.fc1.db),
            ("fc2_W", self.fc2.W, self.fc2.dW),
            ("fc2_b", self.fc2.b, self.fc2.db),
        ]

    def save_weights(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist trained CNN parameters and running BatchNorm stats to an .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: Dict[str, Any] = {name: arr for name, arr, _ in self.param_groups()}
        state.update(
            {
                "bn1_running_mean": self.bn1.running_mean,
                "bn1_running_var": self.bn1.running_var,
                "bn2_running_mean": self.bn2.running_mean,
                "bn2_running_var": self.bn2.running_var,
                "bn3_running_mean": self.bn3.running_mean,
                "bn3_running_var": self.bn3.running_var,
                "num_classes": np.array([self.num_classes], dtype=np.int32),
            }
        )
        if metadata:
            for k, v in metadata.items():
                state[f"meta_{k}"] = np.array([str(v)])
        np.savez_compressed(path, **state)

    @classmethod
    def load_weights(cls, path: Path) -> "FoodCNN":
        """Load a FoodCNN model from an .npz weights file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CNN weights file not found: {path}")
        data = np.load(path, allow_pickle=False)
        num_classes = int(data["num_classes"][0])
        model = cls(num_classes=num_classes)
        for name, arr, _ in model.param_groups():
            arr[...] = data[name]
        model.bn1.running_mean[...] = data["bn1_running_mean"]
        model.bn1.running_var[...] = data["bn1_running_var"]
        model.bn2.running_mean[...] = data["bn2_running_mean"]
        model.bn2.running_var[...] = data["bn2_running_var"]
        model.bn3.running_mean[...] = data["bn3_running_mean"]
        model.bn3.running_var[...] = data["bn3_running_var"]
        return model


class AdamOptimizer:
    """Adam optimizer with optional weight decay for FoodCNN parameters."""

    def __init__(
        self,
        model: FoodCNN,
        lr: float = 3e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {name: np.zeros_like(arr) for name, arr, _ in model.param_groups()}
        self.v: Dict[str, np.ndarray] = {name: np.zeros_like(arr) for name, arr, _ in model.param_groups()}

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for name, param, grad in self.model.param_groups():
            g = grad
            if self.weight_decay > 0.0 and param.ndim >= 2:
                g = g + self.weight_decay * param
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
