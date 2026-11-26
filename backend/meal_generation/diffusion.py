def generate_meal_image(image_prompt, image_model):
    # Send the meal description to the image model.
    result = image_model(image_prompt)
    return result.images[0]
