"""End-to-end tests for Member A's Flask REST API endpoints."""

import base64
import io
from pathlib import Path

import numpy as np

from backend.api_a import create_app
from backend.food_recognition.train_cnn import generate_synthetic_dish_image
from backend.meal_logging.meal_store import MealStore


def _make_png_bytes(dish_key: str = "hamburger") -> bytes:
    img = generate_synthetic_dish_image(dish_key, np.random.default_rng(42))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_end_to_end_workflow(tmp_path: Path):
    store = MealStore(db_path=tmp_path / "api_test.db")
    app = create_app(meal_store=store)
    client = app.test_client()

    # 1. Health
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json()["status"] == "ok"

    # 2. Recognize via multipart image upload
    png_bytes = _make_png_bytes("hamburger")
    rec_resp = client.post(
        "/api/v1/recognize?top_k=3",
        data={"image": (io.BytesIO(png_bytes), "meal.png")},
        content_type="multipart/form-data",
    )
    assert rec_resp.status_code == 200
    rec_data = rec_resp.get_json()
    assert len(rec_data["candidates"]) == 3

    # 3. Recognize via base64 JSON
    b64_str = base64.b64encode(png_bytes).decode("ascii")
    rec_b64_resp = client.post(
        "/api/v1/recognize",
        json={"image_base64": b64_str, "top_k": 2},
    )
    assert rec_b64_resp.status_code == 200
    assert len(rec_b64_resp.get_json()["candidates"]) == 2

    # 4. Nutrition search & calculate
    search_resp = client.get("/api/v1/nutrition/search?q=chicken")
    assert search_resp.status_code == 200
    assert search_resp.get_json()["count"] >= 1

    calc_resp = client.post(
        "/api/v1/nutrition/calculate",
        json={"items": [{"food_name": "Tofu rice bowl", "grams": 380}]},
    )
    assert calc_resp.status_code == 200
    assert calc_resp.get_json()["totals"]["calories_kcal"] > 0

    # 5. Save meal -> List -> Weekly summary -> RAG context -> Delete
    save_resp = client.post(
        "/api/v1/meals",
        json={
            "user_id": "test_user",
            "meal_type": "lunch",
            "eaten_at": "2026-11-26T12:30:00",
            "items": [{"food_name": "Tofu rice bowl", "grams": 380}],
            "recognized_candidates": rec_data["candidates"],
        },
    )
    assert save_resp.status_code == 201
    meal_id = save_resp.get_json()["id"]

    list_resp = client.get("/api/v1/meals?user_id=test_user&days=7")
    assert list_resp.status_code == 200
    assert list_resp.get_json()["count"] == 1

    weekly_resp = client.get("/api/v1/meals/weekly-summary?user_id=test_user&end_date=2026-11-26")
    assert weekly_resp.status_code == 200
    assert weekly_resp.get_json()["total_meals"] == 1

    rag_resp = client.get("/api/v1/meals/rag-context?user_id=test_user&end_date=2026-11-26")
    assert rag_resp.status_code == 200
    assert "Tofu rice bowl" in rag_resp.get_json()["food_log"]

    del_resp = client.delete(f"/api/v1/meals/{meal_id}")
    assert del_resp.status_code == 200
    assert client.get(f"/api/v1/meals/{meal_id}").status_code == 404
