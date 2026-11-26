from openai import OpenAI

client = OpenAI(api_key="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")


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