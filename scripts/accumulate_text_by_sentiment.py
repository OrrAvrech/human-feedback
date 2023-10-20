from pathlib import Path
from utils import read_text
from typing import NamedTuple


class Sentiment(NamedTuple):
    positive = "Positive"
    neutral = "Neutral"
    negative = "Negative"


def main():
    text_dir = Path("/Users/orrav/Documents/Data/human-feedback/raw/text")
    for text_path in text_dir.rglob("*.json"):
        data = read_text(text_path)
        text_segments = [segment["text"] for segment in data][:18]
        sentiments = """Positive
                        Positive
                        Positive
                        Neutral
                        Negative
                        Positive
                        Positive
                        Positive
                        Positive
                        Positive
                        Negative
                        Neutral
                        Negative
                        Negative
                        Negative
                        Positive
                        Positive
                        Positive""".replace(" ", "").split("\n")
        text_corpus = "".join(text_segments)

        samples = []
        end = None
        text_paragraph = text_segments[0]
        accumulated_sentiments = [sentiments[0]]
        start = data[0]["timestamp"][0]
        for i in range(1, len(text_segments)):
            curr_segment = text_segments[i]
            curr_sentiment = sentiments[i]
            prev_sentiment = sentiments[i - 1]

            if (
                curr_sentiment == prev_sentiment
                or curr_sentiment == Sentiment.neutral
            ):
                text_paragraph += curr_segment
                accumulated_sentiments.append(curr_sentiment)
                end = data[i]["timestamp"][-1]
                if end is None:
                    end = data[i]["timestamp"][0]

            else:
                sentiment = Sentiment.positive
                if Sentiment.negative in accumulated_sentiments:
                    sentiment = Sentiment.negative

                sample = {
                    "timestamp": [start, end],
                    "text": text_paragraph,
                    "sentiment": sentiment,
                }
                samples.append(sample)
                start = data[i]["timestamp"][0]
                end = data[i]["timestamp"][-1]
                text_paragraph = text_segments[i]
                accumulated_sentiments = [sentiments[i]]

        if Sentiment.positive in accumulated_sentiments:
            sentiment = Sentiment.positive
        elif Sentiment.negative in accumulated_sentiments:
            sentiment = Sentiment.negative
        else:
            return

        sample = {
            "timestamp": [start, end],
            "text": text_paragraph,
            "sentiment": sentiment,
        }
        samples.append(sample)
        print(samples)


if __name__ == "__main__":
    main()
