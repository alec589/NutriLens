"""High-level Food Recognition service for Member A and downstream consumers (Member C UI)."""

from pathlib import Path
import time
from typing import Optional

import numpy as np

from backend.food_recognition.cnn_model import FoodCNN
from backend.food_recognition.labels import DISH_BY_ID
from backend.food_recognition.preprocessing import ImageInput, preprocess_image
from backend.food_recognition.train_cnn import DEFAULT_WEIGHTS_PATH, train_food_cnn
from backend.meal_logging.nutrition_db import NutritionDB
from backend.meal_logging.schemas import (
    MacroSummary,
    NutritionPer100g,
    RecognitionCandidate,
    RecognitionResponse,
)

DEFAULT_TORCH_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "mobilenet_v2_food.pt"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


class FoodRecognitionService:
    """Loads the best available CNN (MobileNetV2 real-photo CNN or NumPy FoodCNN) and predicts top-K dishes."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        nutrition_db: Optional[NutritionDB] = None,
        auto_train_if_missing: bool = True,
        backend: str = "auto",
    ):
        self.nutrition_db = nutrition_db or NutritionDB()
        self.backend_name = backend

        use_torch = (
            backend == "torch"
            or (backend == "auto" and _torch_available() and DEFAULT_TORCH_WEIGHTS_PATH.exists())
        )

        if use_torch:
            from backend.food_recognition.torch_cnn import TorchFoodCNN

            t_path = Path(weights_path or DEFAULT_TORCH_WEIGHTS_PATH)
            self.torch_model = TorchFoodCNN(weights_path=t_path)
            self.numpy_model: Optional[FoodCNN] = None
            self.backend_name = "torch_mobilenet_v2"
            self.model_name = self.torch_model.MODEL_NAME
            self.num_classes = self.torch_model.num_classes
        else:
            np_path = Path(weights_path or DEFAULT_WEIGHTS_PATH)
            if not np_path.exists():
                if not auto_train_if_missing:
                    raise FileNotFoundError(f"CNN weights not found: {np_path}")
                train_food_cnn(epochs=18, samples_per_class=45, weights_path=np_path, verbose=False)
            self.torch_model = None
            self.numpy_model = FoodCNN.load_weights(np_path)
            self.backend_name = "numpy_cnn"
            self.model_name = FoodCNN.MODEL_NAME
            self.num_classes = self.numpy_model.num_classes

    @property
    def model(self):
        """Compatibility accessor exposing active model metadata."""
        return self.torch_model if self.torch_model is not None else self.numpy_model

    def recognize(self, image_input: ImageInput, top_k: int = 5) -> RecognitionResponse:
        """Predict top-K candidate dish names with confidence and nutrition previews."""
        top_k = max(1, min(top_k, self.num_classes))
        t0 = time.perf_counter()
        if self.torch_model is not None:
            probs = self.torch_model.predict_proba(image_input)[0]
        else:
            assert self.numpy_model is not None
            tensor = preprocess_image(image_input)[None, ...]
            probs = self.numpy_model.predict_proba(tensor)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        top_indices = np.argsort(probs)[::-1][:top_k]
        candidates = []
        for idx in top_indices:
            cid = int(idx)
            dish = DISH_BY_ID[cid]
            entry = self.nutrition_db.get_food(dish.key)
            per_100g: Optional[NutritionPer100g] = None
            default_serving: Optional[MacroSummary] = None
            if entry is not None:
                per_100g = entry.per_100g
                calc = self.nutrition_db.calculate_meal(
                    [{"food_name": dish.key, "grams": dish.default_portion_g}]
                )
                default_serving = calc.totals

            candidates.append(
                RecognitionCandidate(
                    class_id=cid,
                    dish_key=dish.key,
                    name_en=dish.name_en,
                    category=dish.category,
                    confidence=round(float(probs[cid]), 4),
                    default_portion_g=dish.default_portion_g,
                    default_ingredients=list(dish.default_ingredients),
                    nutrition_per_100g=per_100g,
                    nutrition_default_serving=default_serving,
                )
            )

        return RecognitionResponse(
            model_name=self.model_name,
            backend=self.backend_name,
            top_k=top_k,
            inference_time_ms=round(elapsed_ms, 2),
            candidates=candidates,
        )


_default_service: Optional[FoodRecognitionService] = None


def get_recognition_service() -> FoodRecognitionService:
    global _default_service
    if _default_service is None:
        _default_service = FoodRecognitionService()
    return _default_service


def recognize_food(image_input: ImageInput, top_k: int = 5) -> dict:
    """Convenience function returning a JSON-serializable recognition result dict."""
    return get_recognition_service().recognize(image_input, top_k=top_k).model_dump()
