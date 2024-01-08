import os
import typer
import ffmpeg
from shutil import move
from pathlib import Path
from typing import Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pytorchvideo.data.encoded_video import EncodedVideo

app = typer.Typer()


def run_single_vid(input_vid: Path, segment_len: int, output_dir: Path):
    output_vid_dir = output_dir / input_vid.stem
    output_vid_dir.mkdir(exist_ok=True, parents=True)
    output_pattern = str(output_vid_dir / f"{input_vid.stem}_%03d.mp4")
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


def remove_file(filepath: Path):
    os.remove(str(filepath))
    print(f"{filepath.name} removed")


def move_file(filepath: Path, dst_dir: Path):
    move(filepath, dst_dir)
    print(f"{filepath.name} moved to {str(dst_dir)}")


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


@app.command()
def filter_dir(input_dir: Path, duration: Optional[float] = 0, dst_dir: Optional[Path] = None):
    count = 0

    if dst_dir is None:
        filter_action = remove_file
    else:
        dst_dir.mkdir(exist_ok=True, parents=True)
        filter_action = partial(move_file, dst_dir=dst_dir)

    for vid_path in input_dir.rglob("*.mp4"):
        try:
            vid = EncodedVideo.from_path(
                str(vid_path), decode_audio=False, decoder="decord"
            )
            vid_duration = vid.duration
            if vid_duration <= duration:
                filter_action(vid_path)
                print(f"{vid_path.name} with {vid_duration} duration")
                count += 1
        except Exception as e:
            print(f"Unable to load video due to {e}")
            remove_file(vid_path)
            count += 1
    print(f"overall, {count} files have been filtered")


if __name__ == "__main__":
    app()
