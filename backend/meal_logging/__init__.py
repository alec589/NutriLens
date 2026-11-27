"""Meal logging and nutrition module (Member A): nutrition query, calculation, and meal history."""

from backend.meal_logging.nutrition_db import (
    NutritionDB,
    calculate_meal_nutrition,
    get_nutrition_info,
    search_nutrition,
)
from backend.meal_logging.meal_store import (
    MealStore,
    delete_meal,
    get_meal,
    get_rag_food_log,
    get_weekly_summary,
    list_meals,
    save_meal,
    update_meal,
)

__all__ = [
    "NutritionDB",
    "search_nutrition",
    "get_nutrition_info",
    "calculate_meal_nutrition",
    "MealStore",
    "save_meal",
    "get_meal",
    "list_meals",
    "update_meal",
    "delete_meal",
    "get_weekly_summary",
    "get_rag_food_log",
]
