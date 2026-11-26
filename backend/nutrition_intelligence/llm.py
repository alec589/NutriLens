from openai import OpenAI
from config import OPENAI_API_KEY

#import API
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_response(prompt):

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    return response.output_text


if __name__ == "__main__":

    prompt = "What foods are high in protein?"

    answer = generate_response(prompt)

    print(answer)