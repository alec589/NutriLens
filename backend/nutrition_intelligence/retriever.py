from embedding import get_embedding
from knowledge_base import nutrition_knowledge
from sklearn.metrics.pairwise import cosine_similarity

document_embeddings = get_embedding(nutrition_knowledge)

# ensure encapsulation
# choose top 3
def retrieve(question, top_k=3):

    query_embedding = get_embedding(question)

    similarities = cosine_similarity(
        [query_embedding],
        document_embeddings
    )

    scores = similarities[0]

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "text": nutrition_knowledge[index],
            "score": float(scores[index])
        })

    return results


if __name__ == "__main__":

    question = input("Enter your question: ")

    results = retrieve(question)

    for result in results:
        print("Document:", result["text"])
        print("Score:", result["score"])
        print()