"""Internal operations UI for the document teams.

Upload a scanned document, see the OpenCV-cleaned image, the OCR text,
detected layout elements, the predicted document class, and extracted
entities — all served by the FastAPI backend. Kept intentionally simple so
non-technical operators can triage the automated flow.

Run:  streamlit run ui/streamlit_app.py
Env:  DOCAI_API_URL (default http://localhost:8000)
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("DOCAI_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocAI Console", layout="wide")
st.title("📄 DocAI — Document Understanding Console")

with st.sidebar:
    st.header("Backend")
    st.code(API)
    try:
        caps = requests.get(f"{API}/capabilities", timeout=5).json()
        st.success("API online")
        st.json(caps)
    except Exception as e:  # noqa: BLE001
        st.error(f"API unreachable: {e}")

uploaded = st.file_uploader("Upload a scanned document",
                            type=["png", "jpg", "jpeg", "tiff", "tif"])

col1, col2 = st.columns(2)

if uploaded and st.button("Process", type="primary"):
    with st.spinner("Running pipeline…"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        resp = requests.post(f"{API}/process", files=files, timeout=120)

    if resp.status_code != 200:
        st.error(f"Error {resp.status_code}: {resp.text}")
    else:
        result = resp.json()
        with col1:
            st.subheader("Input")
            st.image(uploaded, use_container_width=True)
            if result.get("classification"):
                c = result["classification"]
                st.metric("Predicted class", c["label"], f"{c['score']:.0%}")
            st.caption(f"Timings (ms): {result.get('timings_ms', {})}")

        with col2:
            st.subheader("Extracted text")
            st.text_area("OCR", result.get("text", ""), height=220)

            if result.get("entities"):
                st.subheader("Entities")
                st.dataframe(
                    [{"text": e["text"], "label": e["label"],
                      "score": round(e["score"], 2)}
                     for e in result["entities"]],
                    use_container_width=True,
                )

            if result.get("elements"):
                st.subheader("Detected layout elements")
                st.dataframe(
                    [{"type": el["label"], "score": round(el["score"], 2)}
                     for el in result["elements"]],
                    use_container_width=True,
                )

st.divider()
st.subheader("🔎 Semantic archive search")
q = st.text_input("Search the indexed archive by meaning")
if q:
    try:
        r = requests.post(f"{API}/search", json={"query": q, "k": 5}, timeout=30)
        st.json(r.json())
    except Exception as e:  # noqa: BLE001
        st.warning(f"Search unavailable (build an index first): {e}")
