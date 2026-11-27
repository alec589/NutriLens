"""Pydantic schemas defining Member A's API and service data contracts."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


MealType = Literal["breakfast", "lunch", "dinner", "snack"]
PortionLabel = Literal["small", "medium", "large"]


class NutritionPer100g(BaseModel):
    calories_kcal: float = Field(..., ge=0)
    carbs_g: float = Field(..., ge=0)
    protein_g: float = Field(..., ge=0)
    fat_g: float = Field(..., ge=0)
    fiber_g: float = Field(default=0.0, ge=0)


class FoodNutritionEntry(BaseModel):
    key: str
    name_en: str
    category: str
    aliases: List[str] = Field(default_factory=list)
    default_serving_g: float = Field(..., gt=0)
    serving_unit: str = "1 serving"
    per_100g: NutritionPer100g


class ConfirmedFoodItemInput(BaseModel):
    """User-confirmed food item and portion sent to calculation/save endpoints."""

    food_name: str = Field(..., min_length=1, description="Dish or ingredient name/key")
    grams: Optional[float] = Field(default=None, gt=0, description="Exact weight in grams")
    servings: Optional[float] = Field(default=None, gt=0, description="Number of default servings")
    portion_label: Optional[PortionLabel] = Field(
        default=None,
        description="Optional qualitative portion multiplier: small=0.75x, medium=1.0x, large=1.4x",
    )

    @field_validator("food_name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("food_name cannot be blank")
        return cleaned


class CalculatedFoodItem(BaseModel):
    food_key: str
    name_en: str
    grams: float
    servings: float
    calories_kcal: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float = 0.0
    matched_by: str = "exact"


class MacroSummary(BaseModel):
    calories_kcal: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float = 0.0
    carbs_pct: float = 0.0
    protein_pct: float = 0.0
    fat_pct: float = 0.0


class NutritionCalculationResult(BaseModel):
    items: List[CalculatedFoodItem]
    totals: MacroSummary
    unmatched_items: List[str] = Field(default_factory=list)


class RecognitionCandidate(BaseModel):
    class_id: int
    dish_key: str
    name_en: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    default_portion_g: float
    default_ingredients: List[str] = Field(default_factory=list)
    nutrition_per_100g: Optional[NutritionPer100g] = None
    nutrition_default_serving: Optional[MacroSummary] = None


class RecognitionResponse(BaseModel):
    model_name: str
    backend: str
    top_k: int
    inference_time_ms: float
    candidates: List[RecognitionCandidate]


class SaveMealRequest(BaseModel):
    user_id: str = Field(default="default_user", min_length=1)
    meal_type: MealType = "lunch"
    eaten_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp (e.g. 2026-11-26T12:30:00). Defaults to current local time.",
    )
    items: List[ConfirmedFoodItemInput] = Field(..., min_length=1)
    image_path: Optional[str] = None
    recognized_candidates: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None


class UpdateMealRequest(BaseModel):
    meal_type: Optional[MealType] = None
    eaten_at: Optional[str] = None
    items: Optional[List[ConfirmedFoodItemInput]] = None
    notes: Optional[str] = None


class MealRecord(BaseModel):
    id: str
    user_id: str
    meal_type: MealType
    eaten_at: str
    eaten_date: str
    image_path: Optional[str] = None
    recognized_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    items: List[CalculatedFoodItem]
    totals: MacroSummary
    notes: Optional[str] = None
    created_at: str
    updated_at: str


class DailyNutritionSummary(BaseModel):
    date: str
    meal_count: int
    calories_kcal: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float
    meals: List[Dict[str, Any]] = Field(default_factory=list)


class WeeklyNutritionSummary(BaseModel):
    user_id: str
    start_date: str
    end_date: str
    days_count: int
    logged_days: int
    total_meals: int
    weekly_totals: MacroSummary
    daily_average: MacroSummary
    daily_breakdown: List[DailyNutritionSummary]
    top_dishes: List[Dict[str, Any]]
    insights: List[str]
