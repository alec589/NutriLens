recommendation = {
    "meal_name": "Tofu rice bowl",
    "ingredients": ["tofu", "rice", "broccoli"],
    "nutrition_advice": "Serve the tofu and rice with broccoli for extra fiber.",
    "image_prompt": "An overhead photo of a tofu rice bowl with broccoli.",
}

# Read the meal name from the recommendation.
meal_name = recommendation["meal_name"]
print(meal_name)

# Display each ingredient on its own line.
print("Ingredients:")
for ingredient in recommendation["ingredients"]:
    print("-", ingredient)

# Display the nutrition advice.
print("Nutrition advice:", recommendation["nutrition_advice"])

# Read the description for the meal image.
image_prompt = recommendation["image_prompt"]
print("Image prompt:", image_prompt)
