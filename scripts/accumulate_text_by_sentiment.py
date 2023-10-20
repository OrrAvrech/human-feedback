from pathlib import Path
from utils import read_text


def main():
    sentiment_paragraphs = [
        "Alright, number one is the second upward-facing dog. This is a really, really common one because typically, yogis are a little bit tired, so you notice when I come up into my upward-facing dog, my thighs are touching the ground, my arch is on the way back. Not quite exactly how it should be.",
        "Yeah, so don't do that. Instead, try it this way. When you're going into your forward fold, going through your citation, you come down to your high plank, take your chaturanga, inhale, upward-facing dog. Notice how everything is engaged."
    ]

    text_dir = Path(
        "/Users/orrav/Documents/Data/human-feedback/raw/text"
    )
    for text_path in text_dir.rglob("*.json"):
        data = read_text(text_path)
        text_segments = [segment["text"] for segment in data]
        text_corpus = "".join(text_segments)

        samples = []
        for paragraph in sentiment_paragraphs:
            text_paragraph = ""
            start = 0
            end = None
            for i, current_segment in enumerate(text_segments):
                if current_segment.replace(" ", "") in paragraph.replace(" ", ""):
                    if start == 0:
                        start = data[i]["timestamp"][0]
                    end = data[i]["timestamp"][-1]
                    if end is None:
                        end = data[i]["timestamp"][0]
                    text_paragraph += current_segment

            sample = {"timestamp": [start, end], "text": text_paragraph}
            samples.append(sample)


if __name__ == "__main__":
    main()
