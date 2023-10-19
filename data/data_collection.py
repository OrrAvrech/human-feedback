import yt_dlp
import pyrallis
from pathlib import Path
from utils import extract_audio, transcribe_speech
from data.data_config import DataConfig, ScraperConfig


def scrape_videos(cfg: ScraperConfig, action: str, dataset_dir: Path, video_prefix: str = "video"):
    def filter_videos(info_dict):
        duration = info_dict.get('duration')
        if duration and (duration < cfg.min_vid_duration or duration > cfg.max_vid_duration):
            return "The video is either too short or too long"

    prompt = cfg.prefix_prompt + action
    ydl_opts = {
        "restrictfilenames": cfg.restrict_filenames,
        "match_filter": filter_videos,
        "format": cfg.ext,
        "noplaylist": cfg.no_playlist,
        "quiet": cfg.quiet_mode,
        "outtmpl": {"default": f"{dataset_dir / action / video_prefix}/%(title)s.%(ext)s"},
    }

    agg_duration = cfg.desired_agg_duration
    max_num_urls = agg_duration // cfg.min_vid_duration

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error = ydl.download(f"{cfg.extractor}{max_num_urls}:{prompt}")
        print(error)


@pyrallis.wrap()
def main(cfg: DataConfig):
    # dataset_dir = Path("./dataset")
    dataset_dir = Path("/Users/orrav/Documents/Data/human-feedback/raw")

    # actions = cfg.actions
    # for action in actions:
    #     print(f"{action}:")
    #     scrape_videos(cfg=cfg.scraper, action=action, dataset_dir=dataset_dir)

    # extract audio and transcription from videos
    for vid_path in dataset_dir.rglob("*.mp4"):
        audio_path = extract_audio(vid_path)
        transcribe_speech(audio_path, cfg.transcriber.chunk_length_s)
        # TODO: visualize transcribed text on videos


if __name__ == "__main__":
    main()
