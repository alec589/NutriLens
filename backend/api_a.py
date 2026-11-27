"""Flask REST API for NutriLens Member A (Food Recognition, Nutrition Lookup & Meal Logging)."""

import base64
import io
import os
from typing import Any, Optional

from flask import Flask, jsonify, request
from pydantic import ValidationError

from backend.food_recognition.service import FoodRecognitionService, get_recognition_service
from backend.meal_logging.meal_store import MealStore, get_meal_store
from backend.meal_logging.nutrition_db import NutritionDB, get_nutrition_db
from backend.meal_logging.schemas import ConfirmedFoodItemInput, SaveMealRequest, UpdateMealRequest


def create_app(
    recognition_service: Optional[FoodRecognitionService] = None,
    nutrition_db: Optional[NutritionDB] = None,
    meal_store: Optional[MealStore] = None,
) -> Flask:
    """Create and configure the Member A Flask application."""
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    db = nutrition_db or get_nutrition_db()
    store = meal_store or get_meal_store()
    recognizer = recognition_service or get_recognition_service()

    @app.after_request
    def _add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.errorhandler(ValidationError)
    def _handle_validation_error(err: ValidationError):
        return jsonify({"error": "validation_error", "details": err.errors()}), 400

    @app.errorhandler(ValueError)
    def _handle_value_error(err: ValueError):
        return jsonify({"error": "bad_request", "message": str(err)}), 400

    @app.route("/api/v1/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "nutrilens-member-a",
                "cnn_model": recognizer.model_name,
                "cnn_backend": recognizer.backend_name,
                "num_classes": recognizer.num_classes,
                "nutrition_entries": len(db.list_all()),
                "db_path": str(store.db_path),
            }
        )

    @app.route("/api/v1/recognize", methods=["POST"])
    def recognize():
        top_k_raw = request.args.get("top_k") or request.form.get("top_k")
        image_input: Any = None

        if "image" in request.files or "file" in request.files:
            file_storage = request.files.get("image") or request.files.get("file")
            if file_storage is None or not file_storage.filename and file_storage.content_length == 0:
                return jsonify({"error": "bad_request", "message": "Empty image upload"}), 400
            image_input = io.BytesIO(file_storage.read())
        else:
            payload = request.get_json(silent=True) or {}
            if top_k_raw is None and "top_k" in payload:
                top_k_raw = payload["top_k"]
            if "image_base64" in payload and payload["image_base64"]:
                b64_str = str(payload["image_base64"])
                if "," in b64_str and b64_str.startswith("data:"):
                    b64_str = b64_str.split(",", 1)[1]
                image_input = base64.b64decode(b64_str)
            elif "image_path" in payload and payload["image_path"]:
                image_input = str(payload["image_path"])

        if image_input is None:
            return (
                jsonify(
                    {
                        "error": "bad_request",
                        "message": "Provide multipart 'image'/'file', or JSON 'image_base64' / 'image_path'.",
                    }
                ),
                400,
            )

        try:
            top_k = int(top_k_raw) if top_k_raw is not None else 5
        except (TypeError, ValueError):
            return jsonify({"error": "bad_request", "message": "top_k must be an integer"}), 400

        try:
            result = recognizer.recognize(image_input, top_k=top_k)
        except FileNotFoundError as exc:
            return jsonify({"error": "not_found", "message": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": "recognition_failed", "message": str(exc)}), 400

        return jsonify(result.model_dump())

    @app.route("/api/v1/nutrition/search", methods=["GET"])
    def nutrition_search():
        query = request.args.get("q", "")
        limit_raw = request.args.get("limit", "10")
        try:
            limit = max(1, min(int(limit_raw), 100))
        except ValueError:
            return jsonify({"error": "bad_request", "message": "limit must be an integer"}), 400
        matches = [entry.model_dump() for entry in db.search(query, limit=limit)]
        return jsonify({"query": query, "count": len(matches), "results": matches})

    @app.route("/api/v1/nutrition/foods/<path:food_name>", methods=["GET"])
    def nutrition_get_food(food_name: str):
        entry = db.get_food(food_name)
        if entry is None:
            return jsonify({"error": "not_found", "message": f"Food not found: {food_name}"}), 404
        return jsonify(entry.model_dump())

    @app.route("/api/v1/nutrition/calculate", methods=["POST"])
    def nutrition_calculate():
        payload = request.get_json(silent=True) or {}
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            return jsonify({"error": "bad_request", "message": "'items' must be a non-empty list"}), 400
        parsed_items = [ConfirmedFoodItemInput(**item) for item in raw_items]
        result = db.calculate_meal(parsed_items)
        return jsonify(result.model_dump())

    @app.route("/api/v1/meals", methods=["POST"])
    def meals_create():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "bad_request", "message": "JSON body required"}), 400
        req = SaveMealRequest(**payload)
        record = store.save_meal(req)
        return jsonify(record.model_dump()), 201

    @app.route("/api/v1/meals", methods=["GET"])
    def meals_list():
        user_id = request.args.get("user_id", "default_user")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        meal_type = request.args.get("meal_type")
        days_raw = request.args.get("days")
        limit_raw = request.args.get("limit", "100")

        days = int(days_raw) if days_raw is not None else (7 if not start_date and not end_date else None)
        limit = max(1, min(int(limit_raw), 500))
        records = store.list_meals(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            days=days,
            meal_type=meal_type,
            limit=limit,
        )
        return jsonify({"count": len(records), "meals": [r.model_dump() for r in records]})

    @app.route("/api/v1/meals/weekly-summary", methods=["GET"])
    def meals_weekly_summary():
        user_id = request.args.get("user_id", "default_user")
        end_date = request.args.get("end_date")
        days = int(request.args.get("days", "7"))
        summary = store.get_weekly_summary(user_id=user_id, end_date=end_date, days=days)
        return jsonify(summary.model_dump())

    @app.route("/api/v1/meals/rag-context", methods=["GET"])
    def meals_rag_context():
        user_id = request.args.get("user_id", "default_user")
        end_date = request.args.get("end_date")
        days = int(request.args.get("days", "7"))
        food_log = store.format_food_log_for_rag(user_id=user_id, days=days, end_date=end_date)
        summary = store.get_weekly_summary(user_id=user_id, end_date=end_date, days=days)
        return jsonify(
            {
                "user_id": user_id,
                "days": days,
                "food_log": food_log,
                "summary": summary.model_dump(),
            }
        )

    @app.route("/api/v1/meals/<meal_id>", methods=["GET"])
    def meals_get(meal_id: str):
        record = store.get_meal(meal_id)
        if record is None:
            return jsonify({"error": "not_found", "message": f"Meal record not found: {meal_id}"}), 404
        return jsonify(record.model_dump())

    @app.route("/api/v1/meals/<meal_id>", methods=["PUT"])
    def meals_update(meal_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "bad_request", "message": "JSON body required"}), 400
        req = UpdateMealRequest(**payload)
        record = store.update_meal(meal_id, req)
        if record is None:
            return jsonify({"error": "not_found", "message": f"Meal record not found: {meal_id}"}), 404
        return jsonify(record.model_dump())

    @app.route("/api/v1/meals/<meal_id>", methods=["DELETE"])
    def meals_delete(meal_id: str):
        deleted = store.delete_meal(meal_id)
        if not deleted:
            return jsonify({"error": "not_found", "message": f"Meal record not found: {meal_id}"}), 404
        return jsonify({"deleted": True, "id": meal_id})

    return app


def main() -> None:
    host = os.environ.get("NUTRILENS_HOST", "127.0.0.1")
    port = int(os.environ.get("NUTRILENS_PORT", "5001"))
    app = create_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
