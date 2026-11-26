from retriever import retrieve
from llm import generate_response

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

Provide:
1. Nutrition advice
2. A recommended meal

If the information is insufficient, acknowledge the
limitation instead of inventing facts.
"""

    return prompt

#connect rag and llm
def run_rag(question, food_log):

    prompt = build_prompt(question, food_log)

    answer = generate_response(prompt)

    return answer

if __name__ == "__main__":

    food_log = """
    Breakfast: Eggs and toast
    Lunch: Chicken and rice
    """

    question = input("Enter your question: ")

    answer = run_rag(question, food_log)

    print("\nNutriLens Recommendation:\n")

    print(answer)