from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class ScraperConfig:
    run: bool
    extractor: str
    prefix_prompt: str
    restrict_filenames: bool
    min_vid_duration: int
    max_vid_duration: int
    ext: str
    no_playlist: bool
    desired_agg_duration: int
    quiet_mode: bool
    urls: Optional[list]


@dataclass
class AudioExtractor:
    use_cache: bool


@dataclass
class Transcriber:
    use_cache: bool
    chunk_length_s: Optional[int]


@dataclass
class VideoCutter:
    use_cache: bool


@dataclass
class GPT:
    use_cache: bool


@dataclass
class Templates:
    sentiment_prompt_path: Path


@dataclass
class AlphaPose:
    root_dir: Path


@dataclass
class DataConfig:
    dataset_dir: Path
    output_dir: Path
    actions: list[str]
    scraper: ScraperConfig
    audio_extractor: AudioExtractor
    transcriber: Transcriber
    video_cutter: VideoCutter
    gpt: GPT
    alphapose: AlphaPose
    templates: Templates
