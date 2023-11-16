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
    max_num_url: int
    quiet_mode: bool
    write_auto_subs: bool
    write_info_json: bool
    urls: Optional[list]

    def __post_init__(self):
        if len(self.urls) == 0:
            self.urls = None


@dataclass
class AudioExtractor:
    use_cache: bool


@dataclass
class Transcriber:
    use_cache: bool
    chunk_length_s: Optional[int]


@dataclass
class SentenceSegments:
    use_cache: bool
    use_manual_annotations: bool
    manual_results_path: Path


@dataclass
class VideoCutter:
    use_cache: bool


@dataclass
class Templates:
    system_prompt_path: Path
    user_prompt_path: Path


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
    sentence_segments: SentenceSegments
    video_cutter: VideoCutter
    alphapose: AlphaPose
    templates: Templates
