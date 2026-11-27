"""SQLite persistence, weekly aggregation, and RAG log formatting for NutriLens meal records."""

from collections import Counter
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union
import uuid

from backend.meal_logging.nutrition_db import NutritionDB, _compute_macro_percentages, get_nutrition_db
from backend.meal_logging.schemas import (
    CalculatedFoodItem,
    DailyNutritionSummary,
    MacroSummary,
    MealRecord,
    SaveMealRequest,
    UpdateMealRequest,
    WeeklyNutritionSummary,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "nutrilens.db"


def _parse_eaten_at(eaten_at: Optional[str]) -> tuple[str, str]:
    """Return normalized ISO-8601 timestamp string and YYYY-MM-DD date string."""
    if eaten_at is None or not str(eaten_at).strip():
        dt = datetime.now().replace(microsecond=0)
    else:
        raw = str(eaten_at).strip()
        if len(raw) == 10:
            dt = datetime.fromisoformat(f"{raw}T12:00:00")
        else:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None, microsecond=0)
    return dt.isoformat(), dt.date().isoformat()


class MealStore:
    """SQLite-backed store for confirmed meals, daily/weekly summaries, and Member B RAG context."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        nutrition_db: Optional[NutritionDB] = None,
    ):
        env_path = os.environ.get("NUTRILENS_DB_PATH")
        self.db_path = Path(db_path or env_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.nutrition_db = nutrition_db or get_nutrition_db()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meals (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    meal_type TEXT NOT NULL,
                    eaten_at TEXT NOT NULL,
                    eaten_date TEXT NOT NULL,
                    image_path TEXT,
                    recognized_candidates_json TEXT NOT NULL DEFAULT '[]',
                    total_calories REAL NOT NULL,
                    total_carbs_g REAL NOT NULL,
                    total_protein_g REAL NOT NULL,
                    total_fat_g REAL NOT NULL,
                    total_fiber_g REAL NOT NULL DEFAULT 0.0,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_meals_user_date
                ON meals (user_id, eaten_date DESC, eaten_at DESC);

                CREATE TABLE IF NOT EXISTS meal_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meal_id TEXT NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
                    food_key TEXT NOT NULL,
                    name_en TEXT NOT NULL,
                    grams REAL NOT NULL,
                    servings REAL NOT NULL,
                    calories_kcal REAL NOT NULL,
                    carbs_g REAL NOT NULL,
                    protein_g REAL NOT NULL,
                    fat_g REAL NOT NULL,
                    fiber_g REAL NOT NULL DEFAULT 0.0,
                    matched_by TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_meal_items_meal_id
                ON meal_items (meal_id);
                """
            )

    def _row_to_record(self, conn: sqlite3.Connection, row: sqlite3.Row) -> MealRecord:
        item_rows = conn.execute(
            "SELECT * FROM meal_items WHERE meal_id = ? ORDER BY id ASC",
            (row["id"],),
        ).fetchall()
        items = [
            CalculatedFoodItem(
                food_key=ir["food_key"],
                name_en=ir["name_en"],
                grams=ir["grams"],
                servings=ir["servings"],
                calories_kcal=ir["calories_kcal"],
                carbs_g=ir["carbs_g"],
                protein_g=ir["protein_g"],
                fat_g=ir["fat_g"],
                fiber_g=ir["fiber_g"],
                matched_by=ir["matched_by"],
            )
            for ir in item_rows
        ]
        c_pct, p_pct, f_pct = _compute_macro_percentages(
            row["total_carbs_g"], row["total_protein_g"], row["total_fat_g"]
        )
        totals = MacroSummary(
            calories_kcal=row["total_calories"],
            carbs_g=row["total_carbs_g"],
            protein_g=row["total_protein_g"],
            fat_g=row["total_fat_g"],
            fiber_g=row["total_fiber_g"],
            carbs_pct=c_pct,
            protein_pct=p_pct,
            fat_pct=f_pct,
        )
        return MealRecord(
            id=row["id"],
            user_id=row["user_id"],
            meal_type=row["meal_type"],
            eaten_at=row["eaten_at"],
            eaten_date=row["eaten_date"],
            image_path=row["image_path"],
            recognized_candidates=json.loads(row["recognized_candidates_json"] or "[]"),
            items=items,
            totals=totals,
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_meal(self, request: Union[SaveMealRequest, dict]) -> MealRecord:
        """Validate confirmed items, compute nutrition totals, and persist meal record."""
        req = request if isinstance(request, SaveMealRequest) else SaveMealRequest(**request)
        calc = self.nutrition_db.calculate_meal(req.items)
        if calc.unmatched_items:
            raise ValueError(f"Unrecognized food items in nutrition DB: {calc.unmatched_items}")

        meal_id = uuid.uuid4().hex[:12]
        eaten_at_iso, eaten_date = _parse_eaten_at(req.eaten_at)
        now_iso = datetime.now().replace(microsecond=0).isoformat()
        candidates_json = json.dumps(req.recognized_candidates or [], ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meals (
                    id, user_id, meal_type, eaten_at, eaten_date, image_path,
                    recognized_candidates_json, total_calories, total_carbs_g,
                    total_protein_g, total_fat_g, total_fiber_g, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meal_id,
                    req.user_id,
                    req.meal_type,
                    eaten_at_iso,
                    eaten_date,
                    req.image_path,
                    candidates_json,
                    calc.totals.calories_kcal,
                    calc.totals.carbs_g,
                    calc.totals.protein_g,
                    calc.totals.fat_g,
                    calc.totals.fiber_g,
                    req.notes,
                    now_iso,
                    now_iso,
                ),
            )
            for item in calc.items:
                conn.execute(
                    """
                    INSERT INTO meal_items (
                        meal_id, food_key, name_en, grams, servings,
                        calories_kcal, carbs_g, protein_g, fat_g, fiber_g, matched_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meal_id,
                        item.food_key,
                        item.name_en,
                        item.grams,
                        item.servings,
                        item.calories_kcal,
                        item.carbs_g,
                        item.protein_g,
                        item.fat_g,
                        item.fiber_g,
                        item.matched_by,
                    ),
                )
            row = conn.execute("SELECT * FROM meals WHERE id = ?", (meal_id,)).fetchone()
            assert row is not None
            return self._row_to_record(conn, row)

    def get_meal(self, meal_id: str) -> Optional[MealRecord]:
        """Fetch a single meal record by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM meals WHERE id = ?", (meal_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_record(conn, row)

    def list_meals(
        self,
        user_id: str = "default_user",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        meal_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MealRecord]:
        """Query meal records filtered by user, date range (or trailing N days), and meal_type."""
        if days is not None and start_date is None and end_date is None:
            end_d = date.today()
            start_d = end_d - timedelta(days=max(days - 1, 0))
            start_date = start_d.isoformat()
            end_date = end_d.isoformat()

        clauses = ["user_id = ?"]
        params: List[Any] = [user_id]
        if start_date is not None:
            clauses.append("eaten_date >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("eaten_date <= ?")
            params.append(end_date)
        if meal_type is not None:
            clauses.append("meal_type = ?")
            params.append(meal_type)

        sql = (
            f"SELECT * FROM meals WHERE {' AND '.join(clauses)} "
            "ORDER BY eaten_at DESC, created_at DESC LIMIT ?"
        )
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [self._row_to_record(conn, r) for r in rows]

    def update_meal(self, meal_id: str, request: Union[UpdateMealRequest, dict]) -> Optional[MealRecord]:
        """Update meal metadata or re-calculate items/portions on an existing meal record."""
        req = request if isinstance(request, UpdateMealRequest) else UpdateMealRequest(**request)
        existing = self.get_meal(meal_id)
        if existing is None:
            return None

        meal_type = req.meal_type or existing.meal_type
        notes = req.notes if req.notes is not None else existing.notes
        if req.eaten_at is not None:
            eaten_at_iso, eaten_date = _parse_eaten_at(req.eaten_at)
        else:
            eaten_at_iso, eaten_date = existing.eaten_at, existing.eaten_date

        now_iso = datetime.now().replace(microsecond=0).isoformat()
        with self._connect() as conn:
            if req.items is not None:
                if not req.items:
                    raise ValueError("Updated meal must contain at least one food item")
                calc = self.nutrition_db.calculate_meal(req.items)
                if calc.unmatched_items:
                    raise ValueError(f"Unrecognized food items in nutrition DB: {calc.unmatched_items}")
                conn.execute("DELETE FROM meal_items WHERE meal_id = ?", (meal_id,))
                for item in calc.items:
                    conn.execute(
                        """
                        INSERT INTO meal_items (
                            meal_id, food_key, name_en, grams, servings,
                            calories_kcal, carbs_g, protein_g, fat_g, fiber_g, matched_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            meal_id,
                            item.food_key,
                            item.name_en,
                            item.grams,
                            item.servings,
                            item.calories_kcal,
                            item.carbs_g,
                            item.protein_g,
                            item.fat_g,
                            item.fiber_g,
                            item.matched_by,
                        ),
                    )
                totals = calc.totals
            else:
                totals = existing.totals

            conn.execute(
                """
                UPDATE meals
                SET meal_type = ?, eaten_at = ?, eaten_date = ?, notes = ?,
                    total_calories = ?, total_carbs_g = ?, total_protein_g = ?,
                    total_fat_g = ?, total_fiber_g = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    meal_type,
                    eaten_at_iso,
                    eaten_date,
                    notes,
                    totals.calories_kcal,
                    totals.carbs_g,
                    totals.protein_g,
                    totals.fat_g,
                    totals.fiber_g,
                    now_iso,
                    meal_id,
                ),
            )
            row = conn.execute("SELECT * FROM meals WHERE id = ?", (meal_id,)).fetchone()
            assert row is not None
            return self._row_to_record(conn, row)

    def delete_meal(self, meal_id: str) -> bool:
        """Delete a meal record by ID. Returns True if a row was removed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM meals WHERE id = ?", (meal_id,))
            return cur.rowcount > 0

    def get_weekly_summary(
        self,
        user_id: str = "default_user",
        end_date: Optional[str] = None,
        days: int = 7,
    ) -> WeeklyNutritionSummary:
        """Aggregate 7-day nutrition totals, daily breakdown, top dishes, and insights for Member C."""
        end_d = date.fromisoformat(end_date) if end_date else date.today()
        start_d = end_d - timedelta(days=max(days - 1, 0))
        meals = self.list_meals(
            user_id=user_id,
            start_date=start_d.isoformat(),
            end_date=end_d.isoformat(),
            limit=500,
        )

        by_date: Dict[str, List[MealRecord]] = {
            (start_d + timedelta(days=i)).isoformat(): [] for i in range(days)
        }
        dish_counter: Counter[str] = Counter()
        dish_labels: Dict[str, str] = {}

        for meal in meals:
            by_date.setdefault(meal.eaten_date, []).append(meal)
            for item in meal.items:
                dish_counter[item.food_key] += 1
                dish_labels[item.food_key] = item.name_en

        daily_breakdown: List[DailyNutritionSummary] = []
        w_cal = w_carb = w_prot = w_fat = w_fib = 0.0
        logged_days = 0

        for day_str in sorted(by_date.keys()):
            day_meals = sorted(by_date[day_str], key=lambda m: m.eaten_at)
            d_cal = round(sum(m.totals.calories_kcal for m in day_meals), 1)
            d_carb = round(sum(m.totals.carbs_g for m in day_meals), 1)
            d_prot = round(sum(m.totals.protein_g for m in day_meals), 1)
            d_fat = round(sum(m.totals.fat_g for m in day_meals), 1)
            d_fib = round(sum(m.totals.fiber_g for m in day_meals), 1)

            if day_meals:
                logged_days += 1
            w_cal += d_cal
            w_carb += d_carb
            w_prot += d_prot
            w_fat += d_fat
            w_fib += d_fib

            daily_breakdown.append(
                DailyNutritionSummary(
                    date=day_str,
                    meal_count=len(day_meals),
                    calories_kcal=d_cal,
                    carbs_g=d_carb,
                    protein_g=d_prot,
                    fat_g=d_fat,
                    fiber_g=d_fib,
                    meals=[
                        {
                            "id": m.id,
                            "meal_type": m.meal_type,
                            "eaten_at": m.eaten_at,
                            "items": [f"{it.name_en} ({it.grams}g)" for it in m.items],
                            "calories_kcal": m.totals.calories_kcal,
                        }
                        for m in day_meals
                    ],
                )
            )

        c_pct, p_pct, f_pct = _compute_macro_percentages(w_carb, w_prot, w_fat)
        weekly_totals = MacroSummary(
            calories_kcal=round(w_cal, 1),
            carbs_g=round(w_carb, 1),
            protein_g=round(w_prot, 1),
            fat_g=round(w_fat, 1),
            fiber_g=round(w_fib, 1),
            carbs_pct=c_pct,
            protein_pct=p_pct,
            fat_pct=f_pct,
        )

        denom = max(logged_days, 1)
        daily_average = MacroSummary(
            calories_kcal=round(w_cal / denom, 1) if logged_days else 0.0,
            carbs_g=round(w_carb / denom, 1) if logged_days else 0.0,
            protein_g=round(w_prot / denom, 1) if logged_days else 0.0,
            fat_g=round(w_fat / denom, 1) if logged_days else 0.0,
            fiber_g=round(w_fib / denom, 1) if logged_days else 0.0,
            carbs_pct=c_pct,
            protein_pct=p_pct,
            fat_pct=f_pct,
        )

        top_dishes = [
            {
                "food_key": k,
                "name_en": dish_labels[k],
                "count": cnt,
            }
            for k, cnt in dish_counter.most_common(5)
        ]

        insights: List[str] = []
        if logged_days == 0:
            insights.append("No meals logged in this 7-day window yet.")
        else:
            insights.append(
                f"Logged {len(meals)} meal(s) across {logged_days}/{days} days "
                f"(avg {daily_average.calories_kcal} kcal/day)."
            )
            if p_pct < 15.0:
                insights.append("Protein share is relatively low (<15% of calories); consider lean protein next meal.")
            elif p_pct >= 20.0:
                insights.append("Protein intake is well-supported (>=20% of calories).")
            if daily_average.fiber_g < 18.0:
                insights.append("Average daily fiber is below 18g; adding vegetables, oats, or fruit would help.")

        return WeeklyNutritionSummary(
            user_id=user_id,
            start_date=start_d.isoformat(),
            end_date=end_d.isoformat(),
            days_count=days,
            logged_days=logged_days,
            total_meals=len(meals),
            weekly_totals=weekly_totals,
            daily_average=daily_average,
            daily_breakdown=daily_breakdown,
            top_dishes=top_dishes,
            insights=insights,
        )

    def format_food_log_for_rag(
        self,
        user_id: str = "default_user",
        days: int = 7,
        end_date: Optional[str] = None,
    ) -> str:
        """Format recent meals into a concise text log compatible with Member B's `run_rag(question, food_log)`."""
        summary = self.get_weekly_summary(user_id=user_id, end_date=end_date, days=days)
        if summary.total_meals == 0:
            return f"No meals logged for user '{user_id}' between {summary.start_date} and {summary.end_date}."

        lines = [
            f"Meal History ({summary.start_date} to {summary.end_date}, {summary.total_meals} meals across {summary.logged_days} days):"
        ]
        meals = self.list_meals(
            user_id=user_id,
            start_date=summary.start_date,
            end_date=summary.end_date,
            limit=200,
        )
        meals_chrono = sorted(meals, key=lambda m: m.eaten_at)
        for m in meals_chrono:
            items_desc = ", ".join(f"{it.name_en} ({it.grams}g)" for it in m.items)
            lines.append(
                f"- {m.eaten_date} {m.meal_type.capitalize()}: {items_desc} | "
                f"{m.totals.calories_kcal} kcal (Carbs {m.totals.carbs_g}g, "
                f"Protein {m.totals.protein_g}g, Fat {m.totals.fat_g}g)"
            )
        avg = summary.daily_average
        lines.append(
            f"Daily Average (logged days): {avg.calories_kcal} kcal | "
            f"Carbs {avg.carbs_g}g ({avg.carbs_pct}%), "
            f"Protein {avg.protein_g}g ({avg.protein_pct}%), "
            f"Fat {avg.fat_g}g ({avg.fat_pct}%), Fiber {avg.fiber_g}g."
        )
        return "\n".join(lines)


_default_store: Optional[MealStore] = None


def get_meal_store() -> MealStore:
    global _default_store
    if _default_store is None:
        _default_store = MealStore()
    return _default_store


def save_meal(request: Union[SaveMealRequest, dict]) -> dict:
    return get_meal_store().save_meal(request).model_dump()


def get_meal(meal_id: str) -> Optional[dict]:
    rec = get_meal_store().get_meal(meal_id)
    return rec.model_dump() if rec is not None else None


def list_meals(
    user_id: str = "default_user",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = 7,
    meal_type: Optional[str] = None,
    limit: int = 100,
) -> List[dict]:
    return [
        r.model_dump()
        for r in get_meal_store().list_meals(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            days=days,
            meal_type=meal_type,
            limit=limit,
        )
    ]


def update_meal(meal_id: str, request: Union[UpdateMealRequest, dict]) -> Optional[dict]:
    rec = get_meal_store().update_meal(meal_id, request)
    return rec.model_dump() if rec is not None else None


def delete_meal(meal_id: str) -> bool:
    return get_meal_store().delete_meal(meal_id)


def get_weekly_summary(
    user_id: str = "default_user",
    end_date: Optional[str] = None,
    days: int = 7,
) -> dict:
    return get_meal_store().get_weekly_summary(user_id=user_id, end_date=end_date, days=days).model_dump()


def get_rag_food_log(
    user_id: str = "default_user",
    days: int = 7,
    end_date: Optional[str] = None,
) -> str:
    return get_meal_store().format_food_log_for_rag(user_id=user_id, days=days, end_date=end_date)
