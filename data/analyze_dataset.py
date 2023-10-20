from pathlib import Path
from utils import read_text
import pandas as pd


def estimate_cost(df: pd.DataFrame, num_prompts: int, buffer: float = 0.2):
    # worst-case in terms of word per second
    max_word_per_sec = df["word_per_sec"].max()
    word_per_hour = max_word_per_sec * 60 * 60
    # 1000 words ~ 750 tokens
    in_tokens_per_hour = word_per_hour * 0.75
    # output prompt is approx. 30% longer
    out_token_per_hour = in_tokens_per_hour * 1.3
    # 0.0015$/1k in_tokens
    price_per_in_tokens_h = in_tokens_per_hour * 0.0015 / 1000
    # 0.002$/1k out_tokens
    price_per_out_tokens_h = out_token_per_hour * 0.002 / 1000

    price_per_single_prompt_h = price_per_in_tokens_h + price_per_out_tokens_h
    # taking a buffer in %, just to be on the safe side...
    price_with_buffer_h = price_per_single_prompt_h * (1 + buffer)
    price_per_hour = price_with_buffer_h * num_prompts
    return price_per_hour


def main():
    text_dir = Path(
        "/Users/orrav/Documents/projects/mia_starter_project/dataset/squats/text"
    )
    csv_path = text_dir.parents[1] / "metadata.csv"
    samples = []
    for text_path in text_dir.rglob("*.json"):
        data = read_text(text_path)
        try:
            text_corpus = "".join([segment["text"] for segment in data])
            last_timestamp = data[-1]["timestamp"]
            speech_duration = (
                last_timestamp[0] if last_timestamp[-1] is None else last_timestamp[-1]
            )
            words = text_corpus.split(" ")
            num_words = len(words)
            word_per_sec = num_words / speech_duration
        except Exception as e:
            print(f"{text_path.name}\n{e}")
            continue
        sample = {
            "text": text_corpus,
            "duration": speech_duration,
            "num_words": num_words,
            "word_per_sec": word_per_sec,
        }
        samples.append(sample)

    df = pd.DataFrame(samples)
    df.to_csv(csv_path)

    cost = estimate_cost(df, num_prompts=1)
    print(f"Estimated cost: {cost}$ per 1 hour of speech")


if __name__ == "__main__":
    main()
