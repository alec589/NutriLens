import argparse
import json

from backend.meal_generation.diffusion import generate_meal_image, load_image_model
from backend.nutrition_intelligence.validator import validate_response

parser = argparse.ArgumentParser()
parser.add_argument("--recommendation", help="Path to a recommendation JSON file")
parser.add_argument("--generate-image", action="store_true")
args = parser.parse_args()

recommendation = {
    "meal_name": "Tofu rice bowl",
    "ingredients": ["tofu", "rice", "broccoli"],
    "nutrition_advice": "Serve the tofu and rice with broccoli for extra fiber.",
    "image_prompt": "An overhead photo of a tofu rice bowl with broccoli.",
}

#read recommendation from JSON file（llm provide）
if args.recommendation:
    try:
        with open(args.recommendation, encoding="utf-8") as file:
            recommendation = validate_response(json.load(file))
    except (OSError, ValueError) as error:
        parser.error(str(error))

# read meal name from recommendation
meal_name = recommendation["meal_name"]
print(meal_name)

# display each ingredient 
print("Ingredients:")
for ingredient in recommendation["ingredients"]:
    print("-", ingredient)

# show nutrition advice
print("Nutrition advice:", recommendation["nutrition_advice"])

# read description for meal image
image_prompt = recommendation["image_prompt"]
print("Image prompt:", image_prompt)


# generate meal image when we need 
if args.generate_image:
    image_model = load_image_model()
    image = generate_meal_image(image_prompt, image_model)
    image.save("meal.png")
    print("Saved meal.png")
