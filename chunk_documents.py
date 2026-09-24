import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents_folder = "documents"

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Read every document
for filename in os.listdir(documents_folder):

    file_path = os.path.join(documents_folder, filename)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split the document into chunks
    chunks = text_splitter.split_text(text)

    print("=" * 60)
    print("DOCUMENT:", filename)
    print("NUMBER OF CHUNKS:", len(chunks))
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i + 1} ---")
        print(chunk)

    print()