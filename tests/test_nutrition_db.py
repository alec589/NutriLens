"""Tests for NutritionDB search, alias matching, and macro calculations."""

from backend.meal_logging.nutrition_db import NutritionDB


def test_search_us_dishes_and_ingredients():
    db = NutritionDB()
    bowl_hits = db.search("tofu", limit=5)
    assert any(e.key == "tofu_rice_bowl" for e in bowl_hits)

    burger_hits = db.search("burger", limit=5)
    assert any(e.key == "hamburger" for e in burger_hits)

    salmon_hits = db.search("salmon", limit=5)
    assert any(e.key == "grilled_salmon" for e in salmon_hits)


def test_get_food_by_alias_and_fuzzy():
    db = NutritionDB()
    entry, matched_by = db.get_food_with_match("cheeseburger")
    assert entry is not None
    assert entry.key == "hamburger"
    assert matched_by == "alias"

    fuzzy_entry, fuzzy_match = db.get_food_with_match("Tofu rice bowls")
    assert fuzzy_entry is not None
    assert fuzzy_entry.key == "tofu_rice_bowl"
    assert fuzzy_match in ("substring", "fuzzy", "exact_en")


def test_calculate_meal_with_grams_servings_and_portion_labels():
    db = NutritionDB()
    result = db.calculate_meal(
        [
            {"food_name": "White rice", "grams": 200},
            {"food_name": "Chicken breast", "servings": 1},  # default 150g
            {"food_name": "Broccoli", "portion_label": "large"},  # 120 * 1.4 = 168g
        ]
    )
    assert result.unmatched_items == []
    assert len(result.items) == 3

    rice_item = result.items[0]
    assert rice_item.grams == 200.0
    assert rice_item.calories_kcal == 260.0  # 130 kcal/100g * 2

    chicken_item = result.items[1]
    assert chicken_item.grams == 150.0
    assert chicken_item.protein_g == 46.5  # 31.0 * 1.5

    broccoli_item = result.items[2]
    assert broccoli_item.grams == 168.0

    expected_cal = round(rice_item.calories_kcal + chicken_item.calories_kcal + broccoli_item.calories_kcal, 1)
    assert result.totals.calories_kcal == expected_cal
    pct_sum = result.totals.carbs_pct + result.totals.protein_pct + result.totals.fat_pct
    assert 99.5 <= pct_sum <= 100.5


def test_calculate_meal_reports_unmatched_items():
    db = NutritionDB()
    result = db.calculate_meal(
        [
            {"food_name": "Egg", "servings": 2},
            {"food_name": "Unicorn Dragon Fruit Smoothie X99", "grams": 100},
        ]
    )
    assert len(result.items) == 1
    assert result.unmatched_items == ["Unicorn Dragon Fruit Smoothie X99"]
