from pathlib import Path
from utils import extract_frames, frames_to_vid


def main():
    # vid_path = Path("/Users/orrav/Documents/Data/human-feedback/segments/video/10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_0.0_5.36.mp4")
    # output_dir = Path("/Users/orrav/Documents/Data/human-feedback/segments/frames")
    # vid_output_dir = output_dir / vid_path.stem
    # vid_output_dir.mkdir(exist_ok=True, parents=True)
    # extract_frames(vid_path, vid_output_dir)

    frames_dir = Path("/Users/orrav/Documents/projects/human-feedback/AlphaPose-master/examples/res/vis")
    out_vid_path = Path("/Users/orrav/Documents/Data/human-feedback/segments/video_pose/10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_0.0_5.36.mp4")
    frames_to_vid(frames_dir, out_vid_path, fps=30)


if __name__ == "__main__":
    main()
