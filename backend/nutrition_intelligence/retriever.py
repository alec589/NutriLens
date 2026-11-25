from embedding import get_embedding
from knowledge_base import nutrition_knowledge

document_embeddings = get_embedding(nutrition_knowledge)

print(document_embeddings.shape)