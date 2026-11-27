"""Nutrition database query and portion-based macro calculation for NutriLens (Member A)."""

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

from backend.meal_logging.schemas import (
    CalculatedFoodItem,
    ConfirmedFoodItemInput,
    FoodNutritionEntry,
    MacroSummary,
    NutritionCalculationResult,
)

DEFAULT_NUTRITION_DB_PATH = Path(__file__).resolve().parent / "data" / "nutrition_db.json"

PORTION_MULTIPLIERS: Dict[str, float] = {
    "small": 0.75,
    "medium": 1.0,
    "large": 1.4,
}


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().replace("-", " ").replace("_", " ").split())


def _compute_macro_percentages(carbs_g: float, protein_g: float, fat_g: float) -> Tuple[float, float, float]:
    carbs_kcal = max(carbs_g, 0.0) * 4.0
    protein_kcal = max(protein_g, 0.0) * 4.0
    fat_kcal = max(fat_g, 0.0) * 9.0
    total_macro_kcal = carbs_kcal + protein_kcal + fat_kcal
    if total_macro_kcal <= 0.0:
        return 0.0, 0.0, 0.0
    return (
        round((carbs_kcal / total_macro_kcal) * 100.0, 1),
        round((protein_kcal / total_macro_kcal) * 100.0, 1),
        round((fat_kcal / total_macro_kcal) * 100.0, 1),
    )


class NutritionDB:
    """In-memory nutrition lookup & calculation engine backed by US nutrition_db.json."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_NUTRITION_DB_PATH)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Nutrition database file not found: {self.db_path}")
        raw_items = json.loads(self.db_path.read_text(encoding="utf-8"))
        self.entries: List[FoodNutritionEntry] = [FoodNutritionEntry(**item) for item in raw_items]
        self._lookup: Dict[str, Tuple[FoodNutritionEntry, str]] = {}

        for entry in self.entries:
            self._lookup[entry.key.lower()] = (entry, "key")
            self._lookup[_norm(entry.key)] = (entry, "key")
            self._lookup[_norm(entry.name_en)] = (entry, "exact_en")
            for alias in entry.aliases:
                alias_key = _norm(alias)
                if alias_key and alias_key not in self._lookup:
                    self._lookup[alias_key] = (entry, "alias")

    def list_all(self) -> List[FoodNutritionEntry]:
        """Return all nutrition database entries."""
        return list(self.entries)

    def get_food_with_match(self, food_name: str) -> Tuple[Optional[FoodNutritionEntry], str]:
        """Resolve a food entry by key, English name, alias, or close fuzzy match."""
        if not food_name or not food_name.strip():
            return None, "unmatched"
        raw_lower = food_name.strip().lower()
        normed = _norm(food_name)
        if raw_lower in self._lookup:
            return self._lookup[raw_lower]
        if normed in self._lookup:
            return self._lookup[normed]

        # Substring fallback (prefer query contained in entry, or whole-word entry contained in query)
        best_sub_entry: Optional[FoodNutritionEntry] = None
        best_sub_score = 0.0
        for entry in self.entries:
            candidates = [entry.key, entry.name_en, *entry.aliases]
            for c in candidates:
                c_norm = _norm(c)
                if not c_norm or not normed:
                    continue
                if normed in c_norm:
                    score = len(normed) / max(len(c_norm), 1)
                    if score > best_sub_score:
                        best_sub_score = score
                        best_sub_entry = entry
                elif c_norm in normed and re.search(rf"(?<![a-z0-9]){re.escape(c_norm)}(?![a-z0-9])", normed):
                    score = len(c_norm) / max(len(normed), 1)
                    if score > best_sub_score:
                        best_sub_score = score
                        best_sub_entry = entry
        if best_sub_entry is not None:
            return best_sub_entry, "substring"

        # Fuzzy similarity fallback
        best_entry: Optional[FoodNutritionEntry] = None
        best_score = 0.0
        for entry in self.entries:
            for c in [entry.key, entry.name_en, *entry.aliases]:
                score = SequenceMatcher(None, normed, _norm(c)).ratio()
                if score > best_score:
                    best_score = score
                    best_entry = entry
        if best_entry is not None and best_score >= 0.72:
            return best_entry, "fuzzy"
        return None, "unmatched"

    def get_food(self, food_name: str) -> Optional[FoodNutritionEntry]:
        """Look up a single food nutrition entry."""
        entry, _ = self.get_food_with_match(food_name)
        return entry

    def search(self, query: str, limit: int = 10) -> List[FoodNutritionEntry]:
        """Search food entries by English name, alias, or category."""
        q = _norm(query)
        if not q:
            return self.entries[:limit]

        scored: List[Tuple[float, FoodNutritionEntry]] = []
        for entry in self.entries:
            texts = [entry.key, entry.name_en, entry.category, *entry.aliases]
            best = 0.0
            for t in texts:
                t_norm = _norm(t)
                if q == t_norm:
                    best = max(best, 1.0)
                elif t_norm.startswith(q):
                    best = max(best, 0.92)
                elif q in t_norm:
                    best = max(best, 0.85)
                else:
                    best = max(best, SequenceMatcher(None, q, t_norm).ratio() * 0.75)
            if best >= 0.45:
                scored.append((best, entry))

        scored.sort(key=lambda pair: (-pair[0], pair[1].name_en))
        return [entry for _, entry in scored[:limit]]

    def calculate_meal(
        self,
        items: Sequence[Union[ConfirmedFoodItemInput, dict]],
    ) -> NutritionCalculationResult:
        """Compute per-item and total calories, carbs, protein, fat, and fiber."""
        calculated_items: List[CalculatedFoodItem] = []
        unmatched: List[str] = []

        total_cal = 0.0
        total_carbs = 0.0
        total_protein = 0.0
        total_fat = 0.0
        total_fiber = 0.0

        for raw_item in items:
            item = (
                raw_item
                if isinstance(raw_item, ConfirmedFoodItemInput)
                else ConfirmedFoodItemInput(**raw_item)
            )
            entry, matched_by = self.get_food_with_match(item.food_name)
            if entry is None:
                unmatched.append(item.food_name)
                continue

            if item.grams is not None:
                grams = float(item.grams)
                servings = grams / entry.default_serving_g
            elif item.servings is not None:
                servings = float(item.servings)
                grams = servings * entry.default_serving_g
            elif item.portion_label is not None:
                servings = PORTION_MULTIPLIERS[item.portion_label]
                grams = servings * entry.default_serving_g
            else:
                servings = 1.0
                grams = entry.default_serving_g

            factor = grams / 100.0
            cal = round(entry.per_100g.calories_kcal * factor, 1)
            carbs = round(entry.per_100g.carbs_g * factor, 1)
            protein = round(entry.per_100g.protein_g * factor, 1)
            fat = round(entry.per_100g.fat_g * factor, 1)
            fiber = round(entry.per_100g.fiber_g * factor, 1)

            calculated_items.append(
                CalculatedFoodItem(
                    food_key=entry.key,
                    name_en=entry.name_en,
                    grams=round(grams, 1),
                    servings=round(servings, 2),
                    calories_kcal=cal,
                    carbs_g=carbs,
                    protein_g=protein,
                    fat_g=fat,
                    fiber_g=fiber,
                    matched_by=matched_by,
                )
            )
            total_cal += cal
            total_carbs += carbs
            total_protein += protein
            total_fat += fat
            total_fiber += fiber

        carbs_pct, protein_pct, fat_pct = _compute_macro_percentages(total_carbs, total_protein, total_fat)
        totals = MacroSummary(
            calories_kcal=round(total_cal, 1),
            carbs_g=round(total_carbs, 1),
            protein_g=round(total_protein, 1),
            fat_g=round(total_fat, 1),
            fiber_g=round(total_fiber, 1),
            carbs_pct=carbs_pct,
            protein_pct=protein_pct,
            fat_pct=fat_pct,
        )
        return NutritionCalculationResult(
            items=calculated_items,
            totals=totals,
            unmatched_items=unmatched,
        )


_default_db: Optional[NutritionDB] = None


def get_nutrition_db() -> NutritionDB:
    global _default_db
    if _default_db is None:
        _default_db = NutritionDB()
    return _default_db


def search_nutrition(query: str, limit: int = 10) -> List[dict]:
    """Convenience function returning matching nutrition entries as dicts."""
    return [entry.model_dump() for entry in get_nutrition_db().search(query, limit=limit)]


def get_nutrition_info(food_name: str) -> Optional[dict]:
    """Convenience function returning a single nutrition entry dict or None."""
    entry = get_nutrition_db().get_food(food_name)
    return entry.model_dump() if entry is not None else None


def calculate_meal_nutrition(items: Sequence[Union[ConfirmedFoodItemInput, dict]]) -> dict:
    """Convenience function calculating meal nutrition and returning a dict."""
    return get_nutrition_db().calculate_meal(items).model_dump()
