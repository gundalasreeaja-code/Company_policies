import os

documents_folder = "documents"

for filename in os.listdir(documents_folder):

    file_path = os.path.join(documents_folder, filename)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    print("=" * 60)
    print("DOCUMENT:", filename)
    print("=" * 60)
    print(text)
    print()