import os
import json
import openai
import yt_dlp
import pyrallis

from pathlib import Path
from typing import NamedTuple
from transformers import pipeline
from moviepy.video.io.VideoFileClip import VideoFileClip

from utils import read_text
from data.data_config import DataConfig, ScraperConfig
import jinja2 as j2

openai.api_key = os.getenv("OPENAI_API_KEY")


class ASRModelZoo(NamedTuple):
    whisper_small = "openai/whisper-small"
    wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


class Sentiment(NamedTuple):
    positive = "Positive"
    neutral = "Neutral"
    negative = "Negative"


def scrape_videos(
    cfg: ScraperConfig, action: str, dataset_dir: Path, video_prefix: str = "video"
):
    def filter_videos(info_dict):
        duration = info_dict.get("duration")
        if duration and (
            duration < cfg.min_vid_duration or duration > cfg.max_vid_duration
        ):
            return "The video is either too short or too long"

    prompt = cfg.prefix_prompt + action
    ydl_opts = {
        "restrictfilenames": cfg.restrict_filenames,
        "match_filter": filter_videos,
        "format": cfg.ext,
        "noplaylist": cfg.no_playlist,
        "quiet": cfg.quiet_mode,
        "outtmpl": {
            "default": f"{dataset_dir / action / video_prefix}/%(title)s.%(ext)s"
        },
    }

    agg_duration = cfg.desired_agg_duration
    max_num_urls = agg_duration // cfg.min_vid_duration

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error = ydl.download(f"{cfg.extractor}{max_num_urls}:{prompt}")
        print(error)


def extract_audio(
    vid_path: Path, cache: bool, prefix: str = "audio", ext: str = "wav"
) -> Path:
    audio_dir = vid_path.parents[1] / prefix
    audio_dir.mkdir(exist_ok=True)
    filepath = audio_dir / f"{vid_path.stem}.{ext}"
    if cache is True and filepath.exists():
        print(f"skip audio-extractor, use local {filepath.name}")
    else:
        with VideoFileClip(str(vid_path)) as clip:
            clip.audio.write_audiofile(filepath)
    return filepath


def transcribe_speech(
    audio_path: Path, chunk_len_s: float, cache: bool, prefix: str = "text"
) -> Path:
    text_dir = audio_path.parents[1] / prefix
    text_dir.mkdir(exist_ok=True)
    filepath = text_dir / f"{audio_path.stem}.json"
    if cache is True and filepath.exists():
        print(f"skip transcriber, use local {filepath.name}")
    else:
        # Load pre-trained ASR model
        transcriber = pipeline(
            "automatic-speech-recognition", model=ASRModelZoo.whisper_small
        )
        transcription = transcriber(
            str(audio_path), return_timestamps=True, chunk_length_s=chunk_len_s
        )
        with open(filepath, "w") as fp:
            json.dump(transcription["chunks"], fp)
    return filepath


def prepare_prompt(text_path: Path, template_path: Path) -> str:
    data = read_text(text_path)
    text_segments = [segment["text"] for segment in data]
    txt = ""
    for i, seg in enumerate(text_segments):
        txt += f"{i + 1}.{seg}\n"

    templates_dir = template_path.parent
    environment = j2.Environment(loader=j2.FileSystemLoader(templates_dir))
    template = environment.get_template(template_path.name)
    sentences = {"sentences": txt}
    prompt = template.render(sentences)
    return prompt


def write_gpt_response(prompt: str, output_path: Path, cache: bool):
    if cache is True and output_path.exists():
        print(f"skip ChatGPT, use local {output_path.name}")
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
        )
        with open(output_path, "w") as fp:
            json.dump(response, fp)


def get_gpt_sentiments(gpt_path: Path) -> list[str]:
    response = read_text(gpt_path)
    sentiments = response[0]["choices"]["text"].split("\n")
    return sentiments


def accumulate_text_by_sentiment(text_path: Path, sentiments: list[str]) -> list[dict]:
    data = read_text(text_path)
    text_segments = [segment["text"] for segment in data]
    samples = []
    end = None
    text_paragraph = text_segments[0]
    accumulated_sentiments = [sentiments[0]]
    start = data[0]["timestamp"][0]
    for i in range(1, len(text_segments)):
        curr_segment = text_segments[i]
        curr_sentiment = sentiments[i]
        prev_sentiment = sentiments[i - 1]

        if curr_sentiment == prev_sentiment or curr_sentiment == Sentiment.neutral:
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
        sentiment = Sentiment.neutral
        print(f"all sentences are {sentiment}")

    sample = {
        "timestamp": [start, end],
        "text": text_paragraph,
        "sentiment": sentiment,
    }
    samples.append(sample)
    return samples


def cut_video_by_text_chunks(vid_path: Path, chunks: list[dict], output_dir: Path):
    with VideoFileClip(str(vid_path)) as vid:
        for sentence in chunks:
            start, end = sentence["timestamp"]
            sub_vid = vid.subclip(start, end)
            segment_name = f"{vid_path.stem}_{start}_{end}"
            vid_segment_path = output_dir / "video" / f"{segment_name}.mp4"
            text_segment_path = output_dir / "text" / f"{segment_name}.json"

            sub_vid.write_videofile(str(vid_segment_path))
            with open(text_segment_path, "w") as fp:
                json.dump(sentence, fp)
        sub_vid.close()
    vid.close()


@pyrallis.wrap()
def main(cfg: DataConfig):
    dataset_dir = cfg.dataset_dir
    actions = cfg.actions
    # scrape new videos or use local videos otherwise
    if cfg.scraper.run is True:
        for action in actions:
            print(f"{action}:")
            scrape_videos(cfg=cfg.scraper, action=action, dataset_dir=dataset_dir)

    for vid_path in dataset_dir.rglob("*.mp4"):
        # extract audio and transcription from videos
        audio_path = extract_audio(vid_path, cache=cfg.audio_extractor.use_cache)
        text_path = transcribe_speech(
            audio_path, cfg.transcriber.chunk_length_s, cache=cfg.transcriber.use_cache
        )
        prompt = prepare_prompt(text_path, cfg.templates.sentiment_prompt_path)
        # OPENAI GPT API Call
        gpt_path = cfg.output_dir / "gpt" / text_path.name
        write_gpt_response(prompt, gpt_path, cache=cfg.gpt.use_cache)
        sentiments = get_gpt_sentiments(gpt_path)
        chunks = accumulate_text_by_sentiment(text_path, sentiments)
        cut_video_by_text_chunks(vid_path, chunks, cfg.output_dir)
        # run pose


if __name__ == "__main__":
    main()
