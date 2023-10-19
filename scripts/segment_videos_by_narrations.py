import json
from pathlib import Path
from moviepy.editor import VideoFileClip
from utils import read_text


def main():
    vid_path = Path(
        "/Users/orrav/Documents/Data/human-feedback/raw/video/10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them.mp4"
    )
    text_path = Path(
        "/Users/orrav/Documents/Data/human-feedback/raw/text/10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them.json"
    )
    segments_dir = Path("/Users/orrav/Documents/Data/human-feedback/segments")

    vid_segments_dir = segments_dir / "video"
    text_segments_dir = segments_dir / "text"
    vid_segments_dir.mkdir(parents=True, exist_ok=True)
    text_segments_dir.mkdir(parents=True, exist_ok=True)

    narrations = read_text(text_path)

    with VideoFileClip(str(vid_path)) as vid:
        for sentence in narrations:
            start, end = sentence["timestamp"]
            text = sentence["text"]
            sub_vid = vid.subclip(start, end)
            segment_name = f"{vid_path.stem}_{start}_{end}"
            vid_segment_path = vid_segments_dir / f"{segment_name}.mp4"
            text_segment_path = text_segments_dir / f"{segment_name}.json"

            sub_vid.write_videofile(str(vid_segment_path))
            with open(text_segment_path, "w") as fp:
                json.dump(text, fp)
        sub_vid.close()
    vid.close()


if __name__ == "__main__":
    main()
