import os
import re

RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/corpus.txt"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_corpus():
    all_text = []

    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(RAW_DIR, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            text = clean_text(text)
            all_text.append(text)

    corpus = "\n".join(all_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)

    print("Corpus created!")
    print("Characters:", len(corpus))
    print("Words:", len(corpus.split()))


if __name__ == "__main__":
    build_corpus()