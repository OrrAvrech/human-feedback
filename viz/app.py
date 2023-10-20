import json
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")

video_dir = Path("/Users/orrav/Documents/Data/human-feedback/sentiment_segments/video")
text_dir = Path("/Users/orrav/Documents/Data/human-feedback/sentiment_segments/text/")
segments_list_sorted = sorted(list(video_dir.rglob("*.mp4")), key=lambda x: float(str(x).split("_")[-2]))

add_select = st.sidebar.multiselect(
    "Choose According to Sentiment:",
    ("Positive", "Negative"), ("Positive", "Negative")
)


for i, vid_path in enumerate(segments_list_sorted):
    col1, col2, col3 = st.columns(3, gap="large")

    text_path = vid_path.parents[1] / "text" / f"{vid_path.stem}.json"
    with open(text_path) as fp:
        data = json.load(fp)
        sentiment = data["sentiment"]
    if sentiment not in add_select:
        continue

    with col1:
        if i == 0:
            st.header("Video")
        st.video(str(vid_path))

    with col2:
        if i == 0:
            st.header("Text")
        with open(text_path) as fp:
            data = json.load(fp)
            text = data["text"]
        st.markdown(f"### {text}")

    with col3:
        if i == 0:
            st.header("Sentiment")
        if sentiment == "Positive":
            emoji = ":smile:"
        else:
            emoji = ":worried:"
        st.markdown(f"### {sentiment} {emoji}")

    # with col3:
    #     if i == 0:
    #         st.header("Pose")
    #     pose_path = vid_path.parents[1] / "video_pose" / vid_path.stem / f"h264_AlphaPose_{vid_path.stem}.mp4"
    #     st.video(str(pose_path))
