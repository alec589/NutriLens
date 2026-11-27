import streamlit as st
from backend.meal_generation.diffusion import generate_meal_image, load_image_model


recommendation = {
    "meal_name": "Tofu rice bowl",
    "ingredients": ["tofu", "rice", "broccoli"],
    "nutrition_advice": "Serve the tofu and rice with broccoli for extra fiber.",
    "image_prompt": "An overhead photo of a tofu rice bowl with broccoli.",
}

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
if st.button("Generate meal image"):
    with st.spinner("Generating meal image..."):
        image_model = load_image_model()
        image = generate_meal_image(recommendation["image_prompt"], image_model)
    st.image(image)
