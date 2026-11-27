"""Seed a realistic 7-day US meal log into NutriLens SQLite store for Member B/C integration demos."""

from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.meal_logging.meal_store import MealStore


def seed_week(user_id: str = "default_user") -> int:
    store = MealStore()
    today = date.today()
    sample_plan = [
        (6, "breakfast", "08:30:00", [{"food_name": "Eggs and toast", "servings": 1}]),
        (6, "lunch", "12:40:00", [{"food_name": "Chicken and rice", "grams": 400}]),
        (5, "breakfast", "08:15:00", [{"food_name": "Oatmeal berry bowl", "grams": 280}]),
        (5, "dinner", "19:00:00", [{"food_name": "Grilled salmon", "grams": 230}, {"food_name": "Broccoli", "grams": 150}]),
        (4, "lunch", "12:20:00", [{"food_name": "Club sandwich", "grams": 250}, {"food_name": "Apple", "servings": 1}]),
        (4, "dinner", "18:50:00", [{"food_name": "Spaghetti bolognese", "grams": 350}]),
        (3, "breakfast", "08:45:00", [{"food_name": "Avocado toast", "grams": 170}, {"food_name": "Cafe latte", "grams": 350}]),
        (3, "lunch", "13:00:00", [{"food_name": "Tofu rice bowl", "grams": 380}]),
        (2, "lunch", "12:30:00", [{"food_name": "Caesar salad", "grams": 260}]),
        (2, "dinner", "19:15:00", [{"food_name": "Tacos", "grams": 240}]),
        (1, "breakfast", "08:20:00", [{"food_name": "Omelette", "grams": 210}, {"food_name": "Whole wheat bread", "servings": 1}]),
        (1, "dinner", "19:10:00", [{"food_name": "Steak", "grams": 280}, {"food_name": "Broccoli", "grams": 140}]),
        (0, "lunch", "12:35:00", [{"food_name": "Tofu rice bowl", "grams": 380}]),
    ]

    count = 0
    for days_ago, meal_type, time_str, items in sample_plan:
        day_str = (today - timedelta(days=days_ago)).isoformat()
        store.save_meal(
            {
                "user_id": user_id,
                "meal_type": meal_type,
                "eaten_at": f"{day_str}T{time_str}",
                "items": items,
                "notes": "Seeded 7-day demo meal",
            }
        )
        count += 1
    return count


if __name__ == "__main__":
    inserted = seed_week()
    print(f"Seeded {inserted} meals into MealStore.")
