import ffmpeg
from pathlib import Path


def main():
    vid_dir = Path("/Users/orrav/Documents/Data/human-feedback/segments/video_pose")
    vcodec = "libx264"

    for vid_path in vid_dir.rglob("*.mp4"):
        (
            ffmpeg.input(str(vid_path))
            .output(str(vid_path.parent / f"h264_{vid_path.name}"), vcodec=vcodec)
            .run()
        )
        print(f"converted {vid_path.name} to {vcodec}")
        vid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
