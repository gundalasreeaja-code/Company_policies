import chromadb
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to existing ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)


# Get the existing collection
collection = client.get_collection(
    name="company_policies"
)


# Ask the user for a question
question = input("Enter your question: ")


# Convert the question into an embedding
question_embedding = model.encode(
    question
).tolist()


# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)


# Display the retrieved chunks
print("\nRelevant information:\n")


for i in range(len(results["documents"][0])):

    print("=" * 60)

    print("RESULT", i + 1)

    print("=" * 60)

    print("Source:", results["metadatas"][0][i]["source"])

    print("Chunk:", results["metadatas"][0][i]["chunk"])

    print("\nText:")

    print(results["documents"][0][i])

    print()