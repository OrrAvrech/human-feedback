from pathlib import Path
from utils import extract_frames


def main():
    vid_path = Path("/Users/orrav/Documents/Data/human-feedback/segments/video")
    output_dir = Path("/Users/orrav/Documents/Data/human-feedback/segments/frames")

    for filepath in vid_path.rglob("*.mp4"):
        vid_output_dir = output_dir / filepath.stem
        vid_output_dir.mkdir(exist_ok=True, parents=True)
        extract_frames(filepath, vid_output_dir)


if __name__ == "__main__":
    main()
