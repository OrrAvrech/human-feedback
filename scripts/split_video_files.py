import typer
import ffmpeg
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

app = typer.Typer()


def run_single_vid(input_vid: Path, segment_len: int, output_dir: Path):
    output_pattern = str(output_dir / f"{input_vid.stem}_%03d.mp4")
    (
        ffmpeg.input(input_vid)
        .output(
            output_pattern,
            segment_time=segment_len,
            codec="copy",
            f="segment",
            reset_timestamps=1,
        )
        .run()
    )


def run_vid_list(video_list: list[Path], segment_len: int, output_dir: Path):
    [run_single_vid(vid, segment_len, output_dir) for vid in video_list]


@app.command()
def single(input_vid: Path, segment_len: int, output_dir: Optional[Path] = Path.cwd()):
    run_single_vid(input_vid, segment_len, output_dir)


@app.command()
def folder(
    input_dir: Path,
    segment_len: int,
    workers: int,
    output_dir: Optional[Path] = Path.cwd(),
):
    video_files = list(input_dir.rglob("*.mp4"))
    videos_per_worker = (len(video_files) + workers - 1) // workers
    sub_lists = [
        video_files[i : i + videos_per_worker]
        for i in range(0, len(video_files), videos_per_worker)
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for sublist in sub_lists:
            executor.submit(run_vid_list, sublist, segment_len, output_dir)


if __name__ == "__main__":
    app()
