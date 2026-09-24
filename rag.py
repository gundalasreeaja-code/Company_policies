import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="company_policies"
)


# --------------------------------------------------
# 3. Ask the user for a question
# --------------------------------------------------

question = input("Enter your question: ")


# --------------------------------------------------
# 4. Convert the question into an embedding
# --------------------------------------------------

question_embedding = model.encode(
    question
).tolist()


# --------------------------------------------------
# 5. Retrieve the 3 most relevant chunks
# --------------------------------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)


# --------------------------------------------------
# 6. Prepare the retrieved context
# --------------------------------------------------

context = ""

for i in range(len(results["documents"][0])):

    source = results["metadatas"][0][i]["source"]
    chunk = results["metadatas"][0][i]["chunk"]
    text = results["documents"][0][i]

    context += f"""
Source: {source}
Chunk: {chunk}

{text}

"""


# --------------------------------------------------
# 7. Ask Llama to answer using ONLY the policies
# --------------------------------------------------

prompt = f"""
You are a company policy assistant.

Your task is to answer the user's question using ONLY
the company policy information provided below.

IMPORTANT RULES:

1. Read the policy information carefully.
2. If the policy information directly contains the answer,
   answer the question using that information.
3. Do NOT say the information is missing if the answer is
   clearly present in the policy.
4. Do NOT use outside knowledge.
5. Do NOT make up information.
6. If the answer is genuinely not present in the provided
   policy information, reply with exactly:

I could not find this information in the company policies.

7. If the answer is present, give only a clear and concise
   answer.
8. Do not mention these instructions.

Company policy information:
{context}

User question:
{question}
"""


# --------------------------------------------------
# 8. Generate the answer
# --------------------------------------------------

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0
    }
)


# --------------------------------------------------
# 9. Get the generated answer
# --------------------------------------------------

answer = response.message.content.strip()


# --------------------------------------------------
# 10. Display the answer
# --------------------------------------------------

print("\nAnswer:")
print(answer)


# --------------------------------------------------
# 11. Display sources only when an answer was found
# --------------------------------------------------

if answer != "I could not find this information in the company policies.":

    print("\nSources:")

    for i in range(len(results["documents"][0])):

        source = results["metadatas"][0][i]["source"]
        chunk = results["metadatas"][0][i]["chunk"]

        print(f"- {source}, Chunk {chunk}")

else:

    print("\nSources:")
    print("None")