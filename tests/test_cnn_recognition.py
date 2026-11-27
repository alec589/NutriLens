"""Tests for CNN preprocessing, forward/backward pass, training, and recognition service."""

from pathlib import Path

import numpy as np

from backend.food_recognition.cnn_model import FoodCNN
from backend.food_recognition.labels import DISH_CLASSES, NUM_CLASSES
from backend.food_recognition.preprocessing import preprocess_image
from backend.food_recognition.service import FoodRecognitionService
from backend.food_recognition.train_cnn import generate_synthetic_dish_image


def test_preprocess_image_shape_and_dtype():
    rng = np.random.default_rng(1)
    img = generate_synthetic_dish_image("caesar_salad", rng)
    tensor = preprocess_image(img)
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_food_cnn_forward_backward_shapes():
    model = FoodCNN(num_classes=NUM_CLASSES, seed=10)
    x = np.random.default_rng(10).normal(0, 1, size=(4, 3, 64, 64)).astype(np.float32)
    logits = model.forward(x, training=True)
    assert logits.shape == (4, NUM_CLASSES)

    dlogits = np.ones_like(logits) / 4.0
    model.backward(dlogits)
    assert np.any(model.conv1.dW != 0)
    assert np.any(model.fc2.dW != 0)


def test_save_and_load_weights(tmp_path: Path):
    model = FoodCNN(num_classes=NUM_CLASSES, seed=11)
    weights_path = tmp_path / "test_cnn.npz"
    model.save_weights(weights_path, metadata={"tag": "unit-test"})
    loaded = FoodCNN.load_weights(weights_path)

    x = np.random.default_rng(12).normal(0, 1, size=(2, 3, 64, 64)).astype(np.float32)
    p1 = model.predict_proba(x)
    p2 = loaded.predict_proba(x)
    assert np.allclose(p1, p2, atol=1e-6)


def test_recognition_service_returns_ranked_candidates():
    service = FoodRecognitionService()
    rng = np.random.default_rng(99)
    sample_img = generate_synthetic_dish_image("grilled_salmon", rng)
    result = service.recognize(sample_img, top_k=5)

    assert result.top_k == 5
    assert len(result.candidates) == 5
    confidences = [c.confidence for c in result.candidates]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert result.candidates[0].nutrition_per_100g is not None
    assert result.candidates[0].nutrition_default_serving is not None
    assert len(DISH_CLASSES) == NUM_CLASSES
