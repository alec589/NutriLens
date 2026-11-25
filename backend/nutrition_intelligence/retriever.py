from embedding import get_embedding
from knowledge_base import nutrition_knowledge
from sklearn.metrics.pairwise import cosine_similarity

document_embeddings = get_embedding(nutrition_knowledge)

question = "What foods are high in protein?"

query_embedding = get_embedding(question)

similarities = cosine_similarity(
    [query_embedding],
    document_embeddings
)

print(similarities)
print(similarities.shape)

# choose top 3
scores = similarities[0]

top_k = 3

top_indices = scores.argsort()[::-1][:top_k]

print(top_indices)

for index in top_indices:
    print("Document:", nutrition_knowledge[index])
    print("Score:", scores[index])
    print()