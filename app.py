"""Pediatric stuttering identification demo (research prototype).

    streamlit run app.py

Upload a .wav/.flac/.mp3/.ogg recording, get window-level disfluency predictions
(type probabilities, approximate timestamps) and a severity-proxy band.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from stutter.engine import load_model, resolve_device  # noqa: E402
from stutter.labels import LABELS  # noqa: E402
from infer import analyze  # noqa: E402

DEFAULT_HF_CKPT = "adarshcpt/pediatric-stutter-wav2vec2"


@st.cache_resource
def resolve_checkpoint(ref: str) -> str:
    """Local path -> use directly;  'hf://owner/repo[:file]' -> download from HF Hub."""
    if not ref.startswith("hf://"):
        return ref
    from huggingface_hub import hf_hub_download
    spec = ref[5:]
    repo, _, fname = spec.partition(":")
    return hf_hub_download(repo, fname or "best.pt", repo_type="model")

st.set_page_config(page_title="Pediatric stuttering identification", page_icon=":material/mic:", layout="wide")

DISCLAIMER = (
    "Research prototype only. Predictions are window-level estimates derived from SEP-28k-style "
    "annotations (approximate %SS / SSI-4-style bands) — **not** a clinical diagnosis. "
    "Do not use for medical decisions; consult a licensed speech-language pathologist."
)


@st.cache_resource
def get_model(checkpoint: str, device: str):
    return load_model(checkpoint, resolve_device(device))


@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes, filename: str, checkpoint: str, device: str):
    model, cfg, thresholds = get_model(checkpoint, device)
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        return analyze(tmp_path, model, cfg, thresholds, resolve_device(device))
    finally:
        tmp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- header
st.title("Pediatric stuttering identification")
st.caption("Wav2Vec 2.0 multi-task model — disfluency type probabilities + severity proxy")
st.warning(DISCLAIMER, icon=":material/health_and_safety:")

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    default_ckpt = f"hf://{DEFAULT_HF_CKPT}:best.pt"  # downloads from HF if not local
    if Path("checkpoints/best.pt").exists():
        default_ckpt = "checkpoints/best.pt"
    checkpoint = st.text_input("Checkpoint path or hf://owner/repo[:file]",
                               value=default_ckpt,
                               help="Local best.pt, or hf://repo:file to fetch from Hugging Face")
    device = st.segmented_control("Device", ["auto", "cpu"], default="auto")

# --------------------------------------------------------------------------- upload
uploaded = st.file_uploader("Upload a speech recording", type=["wav", "flac", "mp3", "ogg", "m4a"])

if uploaded is None:
    st.info("Upload a recording to analyze. WAV at 16 kHz mono gives the best results.", icon=":material/upload_file:")
    st.stop()

try:
    ckpt_resolved = resolve_checkpoint(checkpoint)
except Exception as exc:
    st.error(f"Cannot load checkpoint: {exc}")
    st.stop()

st.audio(uploaded.getvalue(), format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}", autoplay=False)

result_slot = st.container()
with result_slot.skeleton():
    try:
        r = run_inference(uploaded.getvalue(), uploaded.name, ckpt_resolved, device)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

# --------------------------------------------------------------------------- summary
windows = pd.DataFrame(r["windows"])

with result_slot:
    st.subheader("Summary")
    m = st.container(horizontal=True)
    m.metric("Severity band", r["severity_band"].title())
    m.metric("Composite score", f"{r['composite_severity_score']:.0f} / 100")
    m.metric("Disfluent windows", f"{r['percent_disfluent_windows']:.0f}%")
    m.metric("Duration", f"{r['duration_s']:.1f} s")

    counts = {k: v for k, v in r["counts"].items() if v}
    if counts:
        st.subheader("Detected events")
        st.dataframe(pd.DataFrame(r["events"]), hide_index=True, alt="Detected disfluency events")
    else:
        st.success("No disfluency events detected above threshold.")

    st.subheader("Event counts")
    st.bar_chart(pd.Series(r["counts"]), alt="Count of predicted events by disfluency type")

    # window probabilities over time
    st.subheader("Window predictions")
    long_df = windows.melt(id_vars=["start_s", "end_s", "severity"],
                           value_vars=LABELS, var_name="Type", value_name="Probability")
    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("start_s:Q", title="Time (s)"),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Type:N", title="Disfluency type"),
            tooltip=["start_s", "end_s", "Type", alt.Tooltip("Probability:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, alt="Disfluency-type probabilities over time, one line per type")
    st.line_chart(windows.set_index("start_s")["severity"], alt="Per-window severity over time")

    with st.expander("Per-window details"):
        st.dataframe(windows, hide_index=True, alt="Per-window probabilities")

    st.download_button("Download result JSON", data=__import__("json").dumps(r, indent=2),
                       file_name=f"{Path(uploaded.name).stem}_analysis.json", mime="application/json",
                       icon=":material/download:")
