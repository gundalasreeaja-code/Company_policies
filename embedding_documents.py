import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

documents_folder = "documents"

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Store all chunks
all_chunks = []

# Read every document
for filename in os.listdir(documents_folder):

    file_path = os.path.join(documents_folder, filename)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split document into chunks
    chunks = text_splitter.split_text(text)

    # Add chunks to the main list
    all_chunks.extend(chunks)

print("Total number of chunks:", len(all_chunks))

# Generate embeddings
embeddings = model.encode(all_chunks)

print("Embeddings created successfully!")
print("Number of embeddings:", len(embeddings))
print("Size of each embedding:", len(embeddings[0]))