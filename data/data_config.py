from pathlib import Path
from typing import Optional
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


@dataclass
class AudioExtractor:
    run: bool


@dataclass
class Transcriber:
    run: bool
    chunk_length_s: Optional[int]


@dataclass
class Templates:
    sentiment_prompt_path: Path


@dataclass
class DataConfig:
    dataset_dir: Path
    actions: list[str]
    scraper: ScraperConfig
    audio_extractor: AudioExtractor
    transcriber: Transcriber
    templates: Templates
