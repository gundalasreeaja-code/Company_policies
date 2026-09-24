import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# Documents folder
documents_folder = "documents"


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Create persistent ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)


# Create collection
collection = client.get_or_create_collection(
    name="company_policies"
)


# Keep track of total chunks
total_chunks = 0


# Read every document
for filename in os.listdir(documents_folder):

    file_path = os.path.join(
        documents_folder,
        filename
    )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    # Split document into chunks
    chunks = text_splitter.split_text(text)


    # Generate embeddings
    embeddings = model.encode(chunks).tolist()


    # Create IDs
    ids = []

    for i in range(len(chunks)):

        ids.append(
            f"{filename}_{i}"
        )


    # Create metadata
    metadatas = []

    for i in range(len(chunks)):

        metadatas.append({
            "source": filename,
            "chunk": i + 1
        })


    # Store everything in ChromaDB
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )


    total_chunks += len(chunks)


print("Documents stored successfully in ChromaDB!")
print("Total chunks stored:", total_chunks)