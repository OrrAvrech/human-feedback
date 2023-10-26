import json
import yt_dlp
import pyrallis

from pathlib import Path
from typing import NamedTuple
from transformers import pipeline
from moviepy.video.io.VideoFileClip import VideoFileClip

from utils import read_text
from data.data_config import DataConfig, ScraperConfig
import jinja2 as j2


class ASRModelZoo(NamedTuple):
    whisper_small = "openai/whisper-small"
    wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


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
    vid_path: Path, audio_prefix: str = "audio", ext: str = "wav", run: bool = True
) -> Path:
    audio_dir = vid_path.parents[1] / audio_prefix
    audio_dir.mkdir(exist_ok=True)
    filepath = audio_dir / f"{vid_path.stem}.{ext}"
    if run is False and filepath.exists():
        pass
    else:
        with VideoFileClip(str(vid_path)) as clip:
            clip.audio.write_audiofile(filepath)
    return filepath


def transcribe_speech(
    audio_path: Path, chunk_len_s, text_prefix: str = "text", run: bool = True
) -> Path:
    text_dir = audio_path.parents[1] / text_prefix
    text_dir.mkdir(exist_ok=True)
    filepath = text_dir / f"{audio_path.stem}.json"
    if run is False and filepath.exists():
        pass
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


@pyrallis.wrap()
def main(cfg: DataConfig):
    dataset_dir = Path("./dataset")

    actions = cfg.actions
    if cfg.scraper.run is True:
        # scrape new videos, use local video otherwise
        for action in actions:
            print(f"{action}:")
            scrape_videos(cfg=cfg.scraper, action=action, dataset_dir=dataset_dir)

    for vid_path in dataset_dir.rglob("*.mp4"):
        # extract audio and transcription from videos
        audio_path = extract_audio(vid_path, run=cfg.audio_extractor.run)
        text_path = transcribe_speech(
            audio_path, cfg.transcriber.chunk_length_s, run=cfg.transcriber.run
        )
        prompt = prepare_prompt(text_path, cfg.templates.sentiment_prompt_path)


if __name__ == "__main__":
    main()
