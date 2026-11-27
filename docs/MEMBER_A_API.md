# NutriLens Member A API & Integration Guide

Member A provides three core backend deliverables for the team:
1. **Food Recognition API** (CNN-based Top-K candidate dish prediction from meal photos)
2. **Nutrition Lookup & Calculation Interface** (USDA-style US food database lookup and portion-based macro estimation)
3. **Meal Record Persistence & Query Interface** (SQLite storage, 7-day dashboard summary for Member C, and formatted food log context for Member B's RAG pipeline)

---

## 1. Run the Local HTTP API Server

```bash
cd /Users/alec/Desktop/NutriLens
python3 -m backend.api_a
# Listens on http://127.0.0.1:5001 by default
```

---

## 2. Direct Python Module Integration (for Member B & Member C)

```python
from backend.food_recognition import recognize_food
from backend.meal_logging import (
    search_nutrition,
    get_nutrition_info,
    calculate_meal_nutrition,
    save_meal,
    list_meals,
    get_weekly_summary,
    get_rag_food_log,
)

# 1. Predict candidate dishes from a meal photo
rec = recognize_food("meal.jpg", top_k=5)

# 2. Search nutrition DB & calculate macros for confirmed items/portions
foods = search_nutrition("chicken")
calc = calculate_meal_nutrition([
    {"food_name": "Tofu rice bowl", "grams": 380},
    {"food_name": "Broccoli", "grams": 100},
])

# 3. Save a confirmed meal record
saved = save_meal({
    "user_id": "default_user",
    "meal_type": "lunch",
    "items": [{"food_name": "Tofu rice bowl", "grams": 380}],
})

# 4. Get 7-day food log string for Member B's `run_rag(question, food_log)`
food_log = get_rag_food_log(user_id="default_user", days=7)

# 5. Get 7-day summary for Member C's weekly nutrition dashboard
weekly = get_weekly_summary(user_id="default_user", days=7)
```

---

## 3. HTTP REST API Endpoints

### Health Check
- `GET /api/v1/health`

### Food Recognition API
- `POST /api/v1/recognize?top_k=5`
  - Accepts `multipart/form-data` (`image` or `file` field)
  - Or JSON: `{"image_base64": "...", "top_k": 5}` / `{"image_path": "...", "top_k": 5}`
  - Returns `candidates` (`dish_key`, `name_en`, `confidence`, `default_portion_g`, `default_ingredients`, `nutrition_per_100g`, `nutrition_default_serving`).

### Nutrition Lookup & Calculation API
- `GET /api/v1/nutrition/search?q=salmon&limit=10`
- `GET /api/v1/nutrition/foods/<food_name>`
- `POST /api/v1/nutrition/calculate`
  ```json
  {
    "items": [
      {"food_name": "Tofu rice bowl", "grams": 380},
      {"food_name": "Egg", "servings": 2}
    ]
  }
  ```

### Meal Record Save & Read API
- `POST /api/v1/meals` — Calculates calories/macros from confirmed items and saves to SQLite
  ```json
  {
    "user_id": "default_user",
    "meal_type": "lunch",
    "eaten_at": "2026-11-26T12:30:00",
    "items": [
      {"food_name": "Tofu rice bowl", "grams": 380}
    ],
    "notes": "Extra steamed broccoli on the side"
  }
  ```
- `GET /api/v1/meals?user_id=default_user&days=7` — List recent meal records
- `GET /api/v1/meals/<meal_id>` — Fetch a single meal record
- `PUT /api/v1/meals/<meal_id>` — Update items/portions and recalculate macros
- `DELETE /api/v1/meals/<meal_id>` — Delete a meal record
- `GET /api/v1/meals/weekly-summary?user_id=default_user&days=7` — Weekly dashboard data for Member C
- `GET /api/v1/meals/rag-context?user_id=default_user&days=7` — Formatted `food_log` text for Member B
