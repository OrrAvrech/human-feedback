import json
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")

video_dir = Path("/Users/orrav/Documents/projects/mia_starter_project/dataset/yoga/video/")
text_dir = Path("/Users/orrav/Documents/projects/mia_starter_project/dataset/yoga/text/")
segments_dir = Path("/Users/orrav/Documents/Data/human-feedback/segments")
segments_list_sorted = sorted(list(segments_dir.rglob("*.mp4")), key=lambda x: float(str(x).split("_")[-2]))


for i, vid_path in enumerate(segments_list_sorted):
    col1, col2, col3, col4 = st.columns(4, gap="large")

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
            st.header("Motion")
        pose_path = "/Users/orrav/Documents/Data/human-feedback/segments/video_pose/10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_0.0_5.36.mp4"
        st.video(str(pose_path))

    with col4:
        if i == 0:
            st.header("Motion")
        pose_path = "/Users/orrav/Documents/Data/human-feedback/segments/video_pose/10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_0.0_5.36.mp4"
        st.video(str(pose_path))