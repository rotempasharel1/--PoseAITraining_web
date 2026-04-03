
from __future__ import annotations

import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="PoseAITraining", page_icon="🏋️", layout="centered")

from backend import SquatAnalyzer, count_labeled_videos, download_gdrive_folder
from main import (
    MODEL_META_PATH,
    MODEL_PATH,
    ROOT,
    TRAINING_SUMMARY_PATH,
    POSE_GRU_PATH,
    POSE_GRU_META_PATH,
    POSE_TABULAR_PATH,
    POSE_TABULAR_META_PATH,
    is_any_model_ready,
    run_training_pipeline,
    run_pose_training_pipeline,
)

DEFAULT_GDRIVE_LINK = "https://drive.google.com/drive/folders/1CSGl4Y7hiTEJ9UuzwMe562WKmXbcvTAD?usp=drive_link"

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .block-container {
        max-width: 800px;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 18px;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        margin-top: 1rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    html, body, [class*="st-"] {
        color: #f8f9fa !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    div[data-testid="stFileUploader"] > section {
        background-color: rgba(0, 0, 0, 0.28) !important;
        border: 2px dashed rgba(255, 255, 255, 0.35) !important;
        border-radius: 15px !important;
    }

    div[data-testid="stFileUploader"] button {
        color: #000000 !important;
        font-weight: 600 !important;
        background-color: #ffffff !important;
    }

    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 500;
    }

    .result-card {
        padding: 20px;
        border-radius: 16px;
        margin-top: 18px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
    }

    .good-card {
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.35);
    }

    .bad-card {
        background: rgba(255, 152, 0, 0.15);
        border: 1px solid rgba(255, 152, 0, 0.35);
    }

    .warning-card {
        background: rgba(255, 193, 7, 0.16);
        border: 1px solid rgba(255, 193, 7, 0.35);
    }

    .tip-card {
        background: rgba(255,255,255,0.06);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.18);
        height: 100%;
    }

    .quote-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px dashed rgba(255,255,255,0.3);
        margin-top: 28px;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .mini-card {
        background: rgba(255, 255, 255, 0.85);
        color: rgba(0, 0, 0, 0.9);
        padding: 16px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.4);
        margin-top: 14px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    .stat-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_analyzer() -> SquatAnalyzer:
    return SquatAnalyzer()


def render_logo() -> None:
    for candidate in [
        ROOT / "logo.png",
        ROOT / "a_logo_for_poseaitraining_features_a_muscular_ma.png",
        Path("logo.png"),
    ]:
        if candidate.exists():
            _, center, _ = st.columns([1.3, 1, 1.3])
            with center:
                st.image(str(candidate), use_container_width=True)
            return


def _confidence_label_text(level: str) -> str:
    level = (level or "").lower()
    if level == "high":
        return "High reliability"
    if level == "medium":
        return "Medium reliability"
    return "Low reliability / borderline"


def render_result(result: dict) -> None:
    prediction = result.get("prediction", "bad").lower()
    confidence = float(result.get("confidence", 0.0))
    confidence_level = str(result.get("confidence_level", "low")).lower()
    keep_tip = result.get("primary_keep_tip") or "Keep your current control and consistency."
    improve_tip = result.get("primary_improve_tip") or "Focus on cleaner squat mechanics in the full movement."
    agreement = float(result.get("agreement", 0.0))
    model_name = result.get("video_model", "model")
    model_sources = result.get("model_sources", [])

    is_good = prediction == "good"
    title = "✅ Good squat" if is_good else "❌ Needs improvement"

    if confidence_level == "high":
        subtitle = f"Confidence: {confidence:.0%} • High reliability"
        css_class = "good-card" if is_good else "bad-card"
    elif confidence_level == "medium":
        subtitle = f"Confidence: {confidence:.0%} • Medium reliability"
        css_class = "good-card" if is_good else "bad-card"
    else:
        subtitle = f"Confidence: {confidence:.0%} • Borderline result"
        css_class = "warning-card"

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <h2 style="margin:0;">{title}</h2>
            <p style="margin:8px 0 0 0; font-size: 16px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pill_html = ""
    pill_html += f"<span class='stat-pill'>Agreement: {agreement:.0%}</span>"
    pill_html += f"<span class='stat-pill'>Models: {', '.join(model_sources) if model_sources else model_name}</span>"
    if result.get("clips_analyzed"):
        pill_html += f"<span class='stat-pill'>Clips analyzed: {int(result['clips_analyzed'])}</span>"

    st.markdown(pill_html, unsafe_allow_html=True)

    if confidence_level == "low":
        st.warning(
            "This looks like a borderline video. The result may improve if you retrain with more examples "
            "or use a cleaner side-view recording."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="tip-card">
                <h4 style="margin-top:0; color:#72ffc1;">Tip to keep</h4>
                <p style="margin-bottom:0;">{keep_tip}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="tip-card">
                <h4 style="margin-top:0; color:#ffd166;">Tip to improve</h4>
                <p style="margin-bottom:0;">{improve_tip}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    probs = result.get("probabilities", {})
    if probs:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Good probability", f"{float(probs.get('good', 0.0)):.0%}")
        with col_b:
            st.metric("Bad probability", f"{float(probs.get('bad', 0.0)):.0%}")

def cleanup_old_outputs() -> None:
    for path in [
        MODEL_PATH,
        MODEL_META_PATH,
        TRAINING_SUMMARY_PATH,
        POSE_GRU_PATH,
        POSE_GRU_META_PATH,
        POSE_TABULAR_PATH,
        POSE_TABULAR_META_PATH,
    ]:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass


quotes = [
    "The only bad workout is the one that didn't happen.",
    "It never gets easier, you just get stronger.",
    "Don't stop when you're tired. Stop when you're done.",
    "Train the movement, not just the moment.",
    "Strong form is built through consistent reps.",
    "Progress is earned one clean rep at a time.",
    "Small improvements repeated daily become big results.",
    "Discipline builds the strength motivation starts.",
    "Every session is a chance to move better than yesterday.",
    "Good form today becomes power tomorrow.",
    "Confidence is built through practice, not luck.",
    "The body adapts when you stay consistent.",
    "Strength grows where patience and effort meet.",
    "Own the basics and the progress will follow.",
    "A better squat starts with one better rep.",
]

render_logo()
st.title("PoseAITraining")
st.write("Upload your squat video and get one clear result with two short tips.")

if not is_any_model_ready():
    st.info(
        "No trained model was found yet. Train once from your labeled videos, "
        "then the app will switch to upload mode."
    )

    gdrive_link = st.text_input(
        "Google Drive folder link",
        value=DEFAULT_GDRIVE_LINK,
        placeholder="Paste a Drive folder that contains good/ and bad/ videos",
    )

    if st.button("Train model", type="primary", use_container_width=True):
        temp_gdrive = ROOT / "temp_gdrive"
        training_succeeded = False
        try:
            cleanup_old_outputs()

            existing_counts = count_labeled_videos(temp_gdrive) if temp_gdrive.exists() else {}
            if existing_counts.get("total", 0) >= 2:
                st.info(
                    f"Using existing local data: "
                    f"{existing_counts.get('good', 0)} good, "
                    f"{existing_counts.get('bad', 0)} bad videos."
                )
            else:
                if temp_gdrive.exists():
                    shutil.rmtree(temp_gdrive)

                with st.spinner("Downloading videos from Google Drive..."):
                    try:
                        download_gdrive_folder(gdrive_link, str(temp_gdrive))
                    except Exception as dl_exc:
                        counts_check = count_labeled_videos(temp_gdrive) if temp_gdrive.exists() else {}
                        if counts_check.get("total", 0) >= 2:
                            st.warning(
                                f"Download stopped early ({dl_exc}), but enough files were found locally. "
                                f"Training will continue on the downloaded subset."
                            )
                        else:
                            raise RuntimeError(f"Download failed and not enough videos were saved: {dl_exc}")

            counts = count_labeled_videos(temp_gdrive)
            if counts.get("total", 0) == 0:
                raise RuntimeError("No videos were found under folders named good and bad.")
            if counts.get("good", 0) == 0:
                raise RuntimeError("No GOOD videos were found.")
            if counts.get("bad", 0) == 0:
                raise RuntimeError("No BAD videos were found.")

            st.success(f"Found {counts['good']} good videos and {counts['bad']} bad videos.")

            with st.spinner("Training the pose model..."):
                try:
                    pose_path, _pose_acc = run_pose_training_pipeline(temp_gdrive)
                    st.success(f"Pose model saved: {Path(pose_path).name}")
                except Exception:
                    pass

            with st.spinner("Training the video model..."):
                model_path = run_training_pipeline(temp_gdrive)

            if not Path(model_path).exists():
                raise RuntimeError("Training finished, but the model file was not saved correctly.")

            training_succeeded = True
            st.cache_resource.clear()
            st.session_state.pop("last_result", None)
            st.success("Training completed. Switching to upload mode...")
            time.sleep(1.2)
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")
        finally:
            if training_succeeded and temp_gdrive.exists():
                shutil.rmtree(temp_gdrive, ignore_errors=True)
else:
    uploaded_file = st.file_uploader("Upload squat video", type=["mp4", "mov", "avi", "mkv", "webm"])

    if uploaded_file is not None:
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.video(uploaded_file)

        if st.button("Analyze video", type="primary", use_container_width=True):
            temp_path = None
            try:
                analyzer = get_analyzer()
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                with st.spinner("Analyzing video..."):
                    result = analyzer.analyze_video(temp_path)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

if st.session_state.get("last_result"):
    render_result(st.session_state["last_result"])

st.markdown(
    f"""
<div class='quote-box'>
    <h3 style='color: #72ffc1; font-style: italic; margin: 0; font-weight: 500;'>💪 "{random.choice(quotes)}"</h3>
</div>
""",
    unsafe_allow_html=True,
)
