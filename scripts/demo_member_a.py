"""End-to-end CLI demo of Member A's workflow:
1. CNN recognizes candidate dish names from a meal photo
2. User confirms food items & portions -> backend calculates nutrition
3. Backend saves meal record to SQLite
4. Weekly summary (for Member C) & RAG food_log (for Member B) are generated
"""

from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.food_recognition.service import FoodRecognitionService
from backend.food_recognition.train_cnn import generate_synthetic_dish_image
from backend.meal_logging.meal_store import MealStore


def main() -> None:
    sample_real_photo = Path("/tmp/food101_real_24/val/hamburger/hamburger_060.jpg")
    if sample_real_photo.exists():
        demo_photo = sample_real_photo
        photo_desc = str(sample_real_photo)
    else:
        rng = np.random.default_rng(7)
        demo_photo = generate_synthetic_dish_image("hamburger", rng)
        photo_desc = "synthetic hamburger sample"

    recognizer = FoodRecognitionService()
    rec_result = recognizer.recognize(demo_photo, top_k=3)

    print(f"=== 1. CNN Recognition Top-3 Candidates ({recognizer.model_name}, input={photo_desc}) ===")
    for cand in rec_result.candidates:
        print(
            f"- {cand.name_en} | confidence={cand.confidence:.3f} | "
            f"default={cand.default_portion_g}g"
        )

    top_dish = rec_result.candidates[0]
    print(f"\n=== 2. User Confirms '{top_dish.name_en}' ({top_dish.default_portion_g}g) + Broccoli (100g) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MealStore(db_path=Path(tmpdir) / "demo.db")
        saved = store.save_meal(
            {
                "user_id": "demo_user",
                "meal_type": "lunch",
                "items": [
                    {"food_name": top_dish.dish_key, "grams": top_dish.default_portion_g},
                    {"food_name": "broccoli", "grams": 100},
                ],
                "recognized_candidates": [c.model_dump() for c in rec_result.candidates],
            }
        )
        print(
            f"Saved Meal ID={saved.id} | Total: {saved.totals.calories_kcal} kcal "
            f"(Carbs {saved.totals.carbs_g}g, Protein {saved.totals.protein_g}g, Fat {saved.totals.fat_g}g)"
        )

        print("\n=== 3. Weekly Dashboard Summary (for Member C) ===")
        summary = store.get_weekly_summary(user_id="demo_user")
        print(f"Range: {summary.start_date} ~ {summary.end_date} | Meals: {summary.total_meals}")
        print(f"Insights: {summary.insights}")

        print("\n=== 4. Formatted Food Log (for Member B RAG) ===")
        print(store.format_food_log_for_rag(user_id="demo_user"))


if __name__ == "__main__":
    main()
