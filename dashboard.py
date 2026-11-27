import json

import streamlit as st

from backend.nutrition_intelligence.validator import validate_response


recommendation = {
    "meal_name": "Tofu rice bowl",
    "ingredients": ["tofu", "rice", "broccoli"],
    "nutrition_advice": "Serve the tofu and rice with broccoli for extra fiber.",
    "image_prompt": "An overhead photo of a tofu rice bowl with broccoli.",
}

# Read the recommendation from the uploaded JSON file.
uploaded_file = st.file_uploader("Upload recommendation JSON", type="json")
if uploaded_file is not None:
    try:
        recommendation = validate_response(json.load(uploaded_file))
    except ValueError as error:
        st.error(f"Could not read the recommendation: {error}")
        st.stop()

#show the recommended meal name
st.title(recommendation["meal_name"])

# show each ingredient 
st.subheader("Ingredients")
for ingredient in recommendation["ingredients"]:
    st.text(f"• {ingredient}")

# show nutrition advice
st.subheader("Nutrition advice")
st.write(recommendation["nutrition_advice"])

# generate a meal image when we click the button
clicked = st.button("Generate meal image")
if clicked:
    from backend.meal_generation.diffusion import generate_meal_image, load_image_model

    with st.spinner("Generating meal image..."):
        image_model = load_image_model()
        image = generate_meal_image(recommendation["image_prompt"], image_model)
    st.image(image)
