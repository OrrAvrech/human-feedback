from pathlib import Path
from utils import read_text
import pandas as pd


def main():
    text_dir = Path()
    for text_path in text_dir.rglob("*.json"):
        data = read_text(text_path)
        text_corpus = "".join([segment["text"] for segment in data])


if __name__ == "__main__":
    main()
