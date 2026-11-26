# define output schema for diffusion model
def validate_response(result):

    required_fields = {
        "nutrition_advice": str,
        "meal_name": str,
        "ingredients": list,
        "image_prompt": str
    }

    if not isinstance(result, dict):
        raise ValueError("Response must be a dictionary")

    for field, expected_type in required_fields.items():

        if field not in result:
            raise ValueError(f"Missing field: {field}")

        if not isinstance(result[field], expected_type):
            raise ValueError(f"Invalid data type: {field}")

    for ingredient in result["ingredients"]:
        if not isinstance(ingredient, str):
            raise ValueError("Ingredient must be a string")

    return result