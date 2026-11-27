# NutriLens

## Meal Recommendation Dashboard

From the project root, create a Python environment with Streamlit to view recommendations and upload JSON files.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install streamlit
python -m streamlit run dashboard.py
```

The dashboard starts with a sample recommendation. To display a recommendation from the nutrition pipeline, save its result as a JSON file and select **Upload recommendation JSON**. The file should contain one recommendation in this format.

```json
{
  "meal_name": "Tofu rice bowl",
  "ingredients": ["tofu", "rice", "broccoli"],
  "nutrition_advice": "Serve the tofu and rice with broccoli.",
  "image_prompt": "An overhead photo of a tofu rice bowl with broccoli."
}
```

### Optional image generation

To use **Generate meal image** locally, install these dependencies in the same environment and restart the dashboard.

```bash
python -m pip install torch diffusers transformers accelerate
```

The first generation may download the Stable Diffusion weights. The model uses CUDA when available and can take longer on CPU. If you generate images in Colab, install the image dependencies there and keep the local dashboard on Streamlit alone.
