import json
import streamlit as st
from backend.nutrition_intelligence.validator import validate_response


st.set_page_config(page_title="NutriLens", page_icon="🍽️", layout="wide")
st.html("""
    <style>
        .nutrilens-header {
            padding: clamp(1.5rem, 4vw, 3rem);
            margin-bottom: 0.75rem;
            border-radius: 1.25rem;
            background: linear-gradient(120deg, #122d23, #246247);
            color: #f4fcf2;
        }
        .nutrilens-header h1 {
            margin: 0;
            padding: 0;
            font-size: clamp(2.25rem, 5vw, 3.5rem);
            color: #f4fcf2;
        }
        .nutrilens-header p {
            margin: 0.35rem 0 0;
            color: #d2e9d7;
        }
        .nutrilens-preview {
            display: grid;
            min-height: 12rem;
            place-items: center;
            padding: 1rem;
            border: 1px dashed #648b7480;
            border-radius: 0.9rem;
            background: #648b7410;
            text-align: center;
        }
    </style>
    <header class="nutrilens-header">
        <h1>NutriLens</h1>
        <p>Meal recommendation dashboard</p>
    </header>
""")

recommendation = {
    "meal_name": "Tofu rice bowl",
    "ingredients": ["tofu", "rice", "broccoli"],
    "nutrition_advice": "Serve the tofu and rice with broccoli for extra fiber.",
    "image_prompt": "An overhead photo of a tofu rice bowl with broccoli.",
}

# Read a recommendation from the sidebar upload.
with st.sidebar:
    st.header("Recommendation source")
    uploaded_file = st.file_uploader("Upload recommendation JSON", type="json")
    if uploaded_file is None:
        st.caption("Showing the sample recommendation.")
    else:
        try:
            data = json.load(uploaded_file)
            recommendation = validate_response(data)
        except ValueError as error:
            st.error(f"Could not read the recommendation: {error}")
            st.stop()

details, preview = st.columns(2, gap="medium")

with details, st.container(border=True):
    st.header(recommendation["meal_name"])
    st.subheader("Ingredients")
    for ingredient in recommendation["ingredients"]:
        st.text(f"• {ingredient}")

    st.divider()
    st.subheader("Nutrition advice")
    st.write(recommendation["nutrition_advice"])

with preview, st.container(border=True):
    st.header("Meal image")
    image_prompt = recommendation["image_prompt"]
    if st.button("Generate meal image", type="primary"):
        from backend.meal_generation.diffusion import generate_meal_image, load_image_model

        with st.spinner("Generating meal image..."):
            # Load the model once per browser session.
            if "image_model" not in st.session_state:
                st.session_state["image_model"] = load_image_model()
            image_model = st.session_state["image_model"]
            image = generate_meal_image(image_prompt, image_model)
        st.session_state["meal_image"] = (image_prompt, image)

    # Keep showing the saved image as long as the prompt stays the same.
    saved_image = st.session_state.get("meal_image")
    if saved_image is not None and saved_image[0] == image_prompt:
        st.image(saved_image[1])
    else:
        st.html('<div class="nutrilens-preview">Your generated image will appear here.</div>')
