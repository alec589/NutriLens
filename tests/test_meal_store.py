"""Tests for SQLite MealStore CRUD, weekly summary aggregation, and RAG log formatting."""

from pathlib import Path

import pytest

from backend.meal_logging.meal_store import MealStore


def test_save_get_update_delete_meal(tmp_path: Path):
    store = MealStore(db_path=tmp_path / "meals.db")
    created = store.save_meal(
        {
            "user_id": "alice",
            "meal_type": "lunch",
            "eaten_at": "2026-11-26T12:30:00",
            "items": [
                {"food_name": "Tofu rice bowl", "grams": 380},
                {"food_name": "Broccoli", "grams": 100},
            ],
            "notes": "Post-workout lunch",
        }
    )
    assert created.id
    assert created.eaten_date == "2026-11-26"
    assert len(created.items) == 2
    assert created.totals.calories_kcal > 0

    fetched = store.get_meal(created.id)
    assert fetched is not None
    assert fetched.notes == "Post-workout lunch"
    assert fetched.totals.calories_kcal == created.totals.calories_kcal

    updated = store.update_meal(
        created.id,
        {
            "meal_type": "dinner",
            "items": [{"food_name": "Grilled salmon", "grams": 220}],
            "notes": "Updated to dinner",
        },
    )
    assert updated is not None
    assert updated.meal_type == "dinner"
    assert len(updated.items) == 1
    assert updated.items[0].food_key == "grilled_salmon"

    assert store.delete_meal(created.id) is True
    assert store.get_meal(created.id) is None


def test_save_meal_rejects_unknown_food(tmp_path: Path):
    store = MealStore(db_path=tmp_path / "meals.db")
    with pytest.raises(ValueError, match="Unrecognized food items"):
        store.save_meal(
            {
                "meal_type": "snack",
                "items": [{"food_name": "Nonexistent Alien Food", "grams": 100}],
            }
        )


def test_weekly_summary_and_rag_context(tmp_path: Path):
    store = MealStore(db_path=tmp_path / "meals.db")
    store.save_meal(
        {
            "user_id": "bob",
            "meal_type": "breakfast",
            "eaten_at": "2026-11-24T08:30:00",
            "items": [{"food_name": "Eggs and toast", "grams": 190}],
        }
    )
    store.save_meal(
        {
            "user_id": "bob",
            "meal_type": "lunch",
            "eaten_at": "2026-11-26T12:15:00",
            "items": [{"food_name": "Chicken and rice", "grams": 400}],
        }
    )

    summary = store.get_weekly_summary(user_id="bob", end_date="2026-11-26", days=7)
    assert summary.start_date == "2026-11-20"
    assert summary.end_date == "2026-11-26"
    assert len(summary.daily_breakdown) == 7
    assert summary.logged_days == 2
    assert summary.total_meals == 2
    assert summary.weekly_totals.calories_kcal > 900
    assert summary.daily_average.calories_kcal == round(summary.weekly_totals.calories_kcal / 2, 1)

    rag_text = store.format_food_log_for_rag(user_id="bob", end_date="2026-11-26", days=7)
    assert "Eggs and toast" in rag_text
    assert "Chicken and rice" in rag_text
    assert "Daily Average" in rag_text
