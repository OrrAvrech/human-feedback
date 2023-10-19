import json
import cv2
import ffmpeg
from pathlib import Path
from typing import Optional, Union
from moviepy.editor import VideoFileClip
from transformers import pipeline
from typing import NamedTuple


class ASRModelZoo(NamedTuple):
    whisper_small = "openai/whisper-small"
    wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


def extract_audio(vid_path: Path, audio_prefix: str = "audio", ext: str = "wav") -> Path:
    with VideoFileClip(str(vid_path)) as clip:
        audio_dir = vid_path.parents[1] / audio_prefix
        audio_dir.mkdir(exist_ok=True)
        filepath = audio_dir / f"{vid_path.stem}.{ext}"
        clip.audio.write_audiofile(filepath)
    return filepath


def transcribe_speech(audio_path: Path, chunk_len_s, text_prefix: str = "text"):
    # Load pre-trained ASR model
    transcriber = pipeline("automatic-speech-recognition", model=ASRModelZoo.whisper_small)
    transcription = transcriber(str(audio_path), return_timestamps=True, chunk_length_s=chunk_len_s)

    text_dir = audio_path.parents[1] / text_prefix
    text_dir.mkdir(exist_ok=True)
    filepath = text_dir / f"{audio_path.stem}.json"
    with open(filepath, "w") as fp:
        json.dump(transcription["chunks"], fp)


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
    files_list = sorted(
        list(frames_dir.glob("*.png")), key=lambda x: int(str(x.stem).split("_")[-1])
    )
    first_image = cv2.imread(str(files_list[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*frmt)
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for image_path in files_list:
        frame = cv2.imread(str(image_path))
        video.write(frame)

    video.release()


def read_text(text_path: Path) -> list[dict]:
    with open(str(text_path), "r") as fp:
        text = json.load(fp)
    return text
