from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    return model.encode(text)


if __name__ == "__main__":
    sentences = [
        "What foods are high in protein?",
        "Chicken and eggs are rich in protein.",
        "Apples contain dietary fiber."
    ]

    embeddings = get_embedding(sentences)

    print(embeddings)
    print(embeddings.shape)