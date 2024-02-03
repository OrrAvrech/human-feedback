from pathlib import Path
from argparse import ArgumentParser
import streamlit as st


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--num_cols", help="number of columns in grid", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    num_cols = int(num_cols)

    subdirs = [subdir for subdir in data_dir.rglob("*") if subdir.is_dir()]

    choose_vid = st.sidebar.selectbox(
        "Select Folder:",
        ["all"] + [str(vid_dir.relative_to(data_dir)) for vid_dir in subdirs],
        placeholder="Select folder or all...",
    )

    for vid_path in data_dir.rglob("*.mp4"):
        if choose_vid == "all":
            pass
        else:
            if str(vid_path.parent.relative_to(data_dir)) not in choose_vid:
                continue
        
        cols = st.columns(num_cols, gap="large")
        for col in cols:
            with col:
                st.video(str(vid_path))
        


if __name__ == "__main__":
    main()