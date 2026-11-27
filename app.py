import sys
from backend.meal_generation.diffusion import generate_meal_image, load_image_model

recommendation = {
    "meal_name": "Tofu rice bowl",
    "ingredients": ["tofu", "rice", "broccoli"],
    "nutrition_advice": "Serve the tofu and rice with broccoli for extra fiber.",
    "image_prompt": "An overhead photo of a tofu rice bowl with broccoli.",
}

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
if "--generate-image" in sys.argv[1:]:
    image_model = load_image_model()
    image = generate_meal_image(image_prompt, image_model)
    image.save("meal.png")
    print("Saved meal.png")