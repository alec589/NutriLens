import json
import streamlit as st
from backend.nutrition_intelligence.validator import validate_response


recommendation = {
    "meal_name": "Tofu rice bowl",
    "ingredients": ["tofu", "rice", "broccoli"],
    "nutrition_advice": "Serve the tofu and rice with broccoli for extra fiber.",
    "image_prompt": "An overhead photo of a tofu rice bowl with broccoli.",
}

# read the recommendation from JSON file（llm）
uploaded_file = st.file_uploader("Upload recommendation JSON", type="json")
if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)
        recommendation = validate_response(data)
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
image_prompt = recommendation["image_prompt"]
clicked = st.button("Generate meal image")
if clicked:
    from backend.meal_generation.diffusion import generate_meal_image, load_image_model
    with st.spinner("Generating meal image..."):
        # load the model once per browser session
        if "image_model" not in st.session_state:
            st.session_state["image_model"] = load_image_model()
        image_model = st.session_state["image_model"]
        image = generate_meal_image(image_prompt, image_model)
    st.session_state["meal_image"] = (image_prompt, image)

# keep showing the same image when prompt don't change
saved_image = st.session_state.get("meal_image")
if saved_image is not None and saved_image[0] == image_prompt:
    st.image(saved_image[1])
