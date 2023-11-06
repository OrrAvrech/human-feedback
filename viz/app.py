import os
import json
import zipfile

import streamlit as st
from pathlib import Path
from google.cloud import storage


def get_data() -> tuple[Path, Path]:
    local = st.secrets["local"]
    if local is True:
        video_dir = Path(st.secrets["video_dir"])
        text_dir = Path(st.secrets["text_dir"])
    else:
        google_app_creds_path = Path.cwd() / "creds.json"
        with open(google_app_creds_path, "w") as fp:
            json.dump(st.secrets["gcs"], fp)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(google_app_creds_path)
        client = storage.Client()
        bucket = client.get_bucket(st.secrets["bucket_name"])
        blob_name = st.secrets["blob_name"]
        blob = bucket.blob(blob_name)
        blob.download_to_filename(blob_name)
        with zipfile.ZipFile(blob_name) as zp:
            zp.extractall(Path.cwd())
        video_dir = Path.cwd() / "video"
        text_dir = Path.cwd() / "text"
    return video_dir, text_dir


def main():
    st.set_page_config(layout="wide")
    video_dir, text_dir = get_data()
    video_dirs_list = [f for f in video_dir.iterdir() if f.is_dir()]

    add_select = st.sidebar.multiselect(
        "Choose According to Correctness:",
        ("Correct", "Incorrect", "Neutral"),
        ("Correct", "Incorrect", "Neutral"),
    )

    choose_vid = st.sidebar.selectbox(
        "Select Video:",
        ["all"] + [vid_dir.name for vid_dir in video_dirs_list],
        placeholder="Select video or all...",
    )

    filter_word_count = st.sidebar.slider("Filter by word count:", 0, 100, 0)

    vid_header, text_header, sentiment_header, pose_header = [False for _ in range(4)]
    for i, single_vid_dir in enumerate(video_dirs_list):
        if choose_vid == "all":
            pass
        else:
            if single_vid_dir.name not in choose_vid:
                continue

        segments_list_sorted = sorted(
            list(single_vid_dir.glob("*.mp4")),
            key=lambda x: float(str(x).split("_")[-2]),
        )

        for j, vid_path in enumerate(segments_list_sorted):
            col1, col2, col3 = st.columns(3, gap="large")

            text_path = text_dir / single_vid_dir.name / f"{vid_path.stem}.json"
            with open(text_path) as fp:
                data = json.load(fp)
                sentiment = data["sentiment"]
                text = data["text"]
            if sentiment not in add_select:
                continue
            if len(text.split(" ")) <= filter_word_count:
                continue

            with col1:
                if vid_header is False:
                    st.header("Video")
                    vid_header = True
                st.video(str(vid_path))

            with col2:
                if text_header is False:
                    st.header("Text")
                    text_header = True
                with open(text_path) as fp:
                    data = json.load(fp)
                    text = data["text"]
                st.markdown(f"{text}")

            with col3:
                if sentiment_header is False:
                    st.header("Correctness")
                    sentiment_header = True
                if sentiment == "Correct":
                    emoji = ":smile:"
                elif sentiment == "Incorrect":
                    emoji = ":scream:"
                else:
                    emoji = ":neutral_face:"
                st.markdown(f"{sentiment} {emoji}")

            # with col4:
            #     if i == 0:
            #         st.header("Pose")
            #     pose_path = (
            #         vid_path.parents[1]
            #         / "video_pose"
            #         / vid_path.stem
            #         / f"h264_AlphaPose_{vid_path.stem}.mp4"
            #     )
            #     st.video(str(pose_path))
        st.divider()


if __name__ == "__main__":
    main()
