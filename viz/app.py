import json
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")

video_dir = Path("/Users/orrav/Documents/projects/mia_starter_project/dataset/yoga/video/")
text_dir = Path("/Users/orrav/Documents/projects/mia_starter_project/dataset/yoga/text/")
segments_dir = Path("/Users/orrav/Documents/Data/human-feedback/segments")
video_segments_dir = segments_dir / "video"
segments_list_sorted = sorted(list(video_segments_dir.rglob("*.mp4")), key=lambda x: float(str(x).split("_")[-2]))


for i, vid_path in enumerate(segments_list_sorted):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        if i == 0:
            st.header("Video")
        st.video(str(vid_path))

    with col2:
        if i == 0:
            st.header("Text")
        text_path = vid_path.parents[1] / "text" / f"{vid_path.stem}.json"
        with open(text_path) as fp:
            data = json.load(fp)
        st.markdown(f"### {data}")

    with col3:
        if i == 0:
            st.header("Pose")
        pose_path = vid_path.parents[1] / "video_pose" / vid_path.stem / f"h264_AlphaPose_{vid_path.stem}.mp4"
        st.video(str(pose_path))
