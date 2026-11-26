from retriever import retrieve
from llm import generate_response
import json
from validator import validate_response

# construct prompt
def build_prompt(question, food_log):

    results = retrieve(question)

    context = "\n".join(
        result["text"] for result in results
    )

    prompt = f"""
You are a nutrition assistant called NutriLens.

Use the retrieved nutrition knowledge and the user's
food log to answer the question.

User's Food Log:
{food_log}

Retrieved Nutrition Knowledge:
{context}

User Question:
{question}

## ready for diffusion model 
Return your response as a valid JSON object
with the following fields:

- nutrition_advice: A string containing nutrition advice.
- meal_name: A string containing one recommended meal.
- ingredients: A list of ingredient names.
- image_prompt: A detailed visual description of the
  recommended meal for an image generation model.

Recommend only one meal.

Do not include Markdown or code fences.

If the information is insufficient, acknowledge the
limitation instead of inventing facts.

Do not assume portion sizes, calorie intake, allergies,
or dietary restrictions that are not provided.
"""

    return prompt

#connect rag and llm
def run_rag(question, food_log):

    prompt = build_prompt(question, food_log)

    answer = generate_response(prompt)

    result = json.loads(answer)

    result = validate_response(result)

    return result

if __name__ == "__main__":

    food_log = """
    Breakfast: Eggs and toast
    Lunch: Chicken and rice
    """

    question = input("Enter your question: ")

    result = run_rag(question, food_log)

    print("\nNutrition Advice:")
    print(result["nutrition_advice"])

    print("\nRecommended Meal:")
    print(result["meal_name"])

    print("\nIngredients:")
    for ingredient in result["ingredients"]:
        print("-", ingredient)

    print("\nImage Prompt:")
    print(result["image_prompt"])

    print("\nData Type:")
    print(type(result))