import cv2
import ffmpeg
from pathlib import Path
from typing import Optional, Union


def write_lines(file_path: Union[Path, str], lines: list):
    with open(file_path, "w") as file:
        for line in lines:
            file.write(f"file '{line}'" + "\n")


def get_fps(vid_path: Path) -> float:
    probe = ffmpeg.probe(str(vid_path))

    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            original_fps = eval(stream["r_frame_rate"])
            return original_fps


def extract_frames(vid_path: Path, output_dir: Path, fps: Optional[float] = None):
    if fps is None:
        fps = get_fps(vid_path)
    (
        ffmpeg.input(str(vid_path))
        .output(str(output_dir / "frame_%04d.png"), vf=f"fps={fps}")
        .run()
    )
    return fps


def frames_to_vid(frames_dir: Path, output_path: Path, fps: float, frmt: str = "H264"):
    files_list = sorted(list(frames_dir.glob("*.png")), key=lambda x: int(str(x.stem).split("_")[-1]))
    first_image = cv2.imread(str(files_list[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*frmt)
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for image_path in files_list:
        frame = cv2.imread(str(image_path))
        video.write(frame)

    video.release()
