"""Streamlit UI for the resume RAG + job matcher.

Run from the project root:

    streamlit run app.py

Everything heavy lives in the backend modules (resume_rag / job_matcher /
reranker); this app only collects inputs and renders results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

import fs_tools
from job_matcher import JobMatcher
from resume_rag import ResumeRAG

JD_DIR = Path("job_descriptions")
JD_UPLOAD_DIR = JD_DIR / "uploads"
PASTE_OPTION = "(paste your own)"

CORPORA = {
    "Clean (36)": ("resumes", "resumes"),
    "Hard (40)": ("resumes_hard", "resumes_hard"),
}

MODES = ("Hybrid + rerank", "Hybrid", "Semantic only")

BREAKDOWN_BARS = (
    ("semantic", "Semantic similarity"),
    ("keyword_bm25", "Keyword (BM25)"),
    ("rerank", "Cross-encoder rerank"),
    ("skill_coverage", "Required-skill coverage"),
    ("experience_fit", "Experience fit"),
)


# --- cached backend resources ----------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_rag(resumes_dir: str, collection: str) -> ResumeRAG:
    return ResumeRAG(resumes_dir=resumes_dir, collection_name=collection)


@st.cache_resource(show_spinner="Loading cross-encoder model...")
def get_reranker():
    from reranker import CrossEncoderReranker

    return CrossEncoderReranker()


# --- small helpers -----------------------------------------------------------------


def read_text(path: str) -> Tuple[bool, str]:
    """Read a document through the Milestone 1 tools: (ok, text-or-error)."""
    result = fs_tools.read_file(path)
    if result.get("success"):
        return True, str(result.get("content", ""))
    return False, str(result.get("error", "unknown error"))


def jd_file_options() -> List[str]:
    return sorted(p.name for p in JD_DIR.glob("*.txt"))


def save_upload(upload, target_dir: str) -> Tuple[bool, str]:
    """Persist an uploaded txt/pdf/docx under *target_dir*: (ok, path-or-error)."""
    fname = Path(upload.name).name
    if ".." in fname or not fname:
        return False, "invalid filename"
    target = Path(target_dir) / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    if fname.lower().endswith(".txt"):
        text = upload.getvalue().decode("utf-8", errors="replace")
        written = fs_tools.write_file(str(target), text)
        if not written.get("success"):
            return False, str(written.get("error", "write failed"))
    else:
        target.write_bytes(upload.getvalue())
    ok, err = read_text(str(target))
    if not ok or not err.strip():
        target.unlink(missing_ok=True)
        return False, f"could not parse upload: {err if not ok else 'empty document'}"
    return True, str(target)


def ensure_indexed(rag: ResumeRAG) -> None:
    if rag.count() == 0:
        with st.spinner("First use of this corpus — building the index..."):
            stats = rag.build_index(rebuild=True)
        st.toast(f"Indexed {stats.files_indexed} files → {stats.chunks_indexed} chunks")


def bold_section_prefix(excerpt: str) -> str:
    """'[EXPERIENCE] text' -> '**[EXPERIENCE]** text' for markdown rendering."""
    if excerpt.startswith("[") and "]" in excerpt:
        head, _, tail = excerpt.partition("]")
        return f"**{head}]**{tail}"
    return excerpt


def clamp01(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# --- page scaffold -----------------------------------------------------------------

st.set_page_config(page_title="Resume Matcher", page_icon="🎯", layout="wide")
st.title("🎯 Resume Matcher")
st.caption(
    "fs_tools loaders → section-aware chunks → MiniLM embeddings → ChromaDB → "
    "hybrid retrieval (semantic + BM25, optional cross-encoder rerank) → "
    "0-100 scores with must-have filtering."
)

# --- sidebar: corpus, matching knobs, indexing -------------------------------------

with st.sidebar:
    st.header("Corpus")
    corpus_label = st.radio("Resume corpus", list(CORPORA), index=0)
    resumes_dir, collection = CORPORA[corpus_label]

    st.header("Matching")
    k = st.slider("Top-K candidates", min_value=1, max_value=20, value=10)
    mode = st.radio("Retrieval mode", MODES, index=1)
    semantic_weight = st.slider(
        "Semantic weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05,
        help="Semantic share vs BM25 in hybrid modes; ignored in semantic-only.",
    )
    apply_filters = st.checkbox("Apply must-have filters", value=True)

    st.divider()
    st.header("Index")
    use_llm = st.checkbox(
        "Use Claude for tricky resumes (auto)", value=False,
        help="During indexing, low-confidence resumes are re-extracted via the "
             "Anthropic API (cached). Needs ANTHROPIC_API_KEY.",
    )
    if st.button("Rebuild index"):
        os.environ["RESUME_RAG_LLM"] = "auto" if use_llm else "off"
        rag = get_rag(resumes_dir, collection)
        with st.spinner(f"Re-indexing {resumes_dir}/ ..."):
            stats = rag.build_index(rebuild=True)
        st.success(
            f"{stats.files_indexed} files → {stats.chunks_indexed} chunks "
            f"(LLM-assisted: {stats.llm_assisted})"
        )
        st.rerun()

    st.subheader("Add a resume")
    resume_upload = st.file_uploader(
        "Drop a resume into the selected corpus", type=["txt", "pdf", "docx"],
        key="resume_upload",
    )
    if resume_upload is not None and st.button("Index it"):
        ok, info = save_upload(resume_upload, resumes_dir)
        if not ok:
            st.error(info)
        else:
            os.environ["RESUME_RAG_LLM"] = "auto" if use_llm else "off"
            rag = get_rag(resumes_dir, collection)
            before = rag.count()
            with st.spinner("Indexing new resume..."):
                rag.build_index(rebuild=False)
            st.success(f"Indexed {Path(info).name} (+{rag.count() - before} chunks)")
            st.rerun()

# --- main pane: job description input ----------------------------------------------

st.subheader("Job description")

if "jd_text" not in st.session_state:
    default_jd = JD_DIR / "senior_ml_engineer.txt"
    ok, text = read_text(str(default_jd)) if default_jd.exists() else (False, "")
    st.session_state["jd_text"] = text if ok else ""
    st.session_state["_last_jd_choice"] = "senior_ml_engineer.txt"

left, right = st.columns([3, 2])
with left:
    choices = [PASTE_OPTION] + jd_file_options()
    last_choice = st.session_state.get("_last_jd_choice", PASTE_OPTION)
    index = choices.index(last_choice) if last_choice in choices else 0
    selected = st.selectbox("Load a job description", choices, index=index)
    if selected != st.session_state.get("_last_jd_choice"):
        st.session_state["_last_jd_choice"] = selected
        if selected != PASTE_OPTION:
            ok, text = read_text(str(JD_DIR / selected))
            if ok:
                st.session_state["jd_text"] = text
            else:
                st.error(text)
        st.rerun()
with right:
    jd_upload = st.file_uploader(
        "...or upload a JD (txt/pdf/docx)", type=["txt", "pdf", "docx"], key="jd_upload"
    )
    if jd_upload is not None and st.button("Use uploaded JD"):
        ok, info = save_upload(jd_upload, str(JD_UPLOAD_DIR))
        if not ok:
            st.error(info)
        else:
            ok, text = read_text(info)
            if ok:
                st.session_state["jd_text"] = text
                st.session_state["_last_jd_choice"] = PASTE_OPTION
                st.rerun()
            else:
                st.error(text)

jd_text = st.text_area("Job description text", key="jd_text", height=260)

run_match = st.button("Match candidates", type="primary")

# --- matching + results --------------------------------------------------------------

if run_match:
    if not jd_text or not jd_text.strip():
        st.warning("Paste or load a job description first.")
        st.stop()

    rag = get_rag(resumes_dir, collection)
    ensure_indexed(rag)

    use_rerank = mode == "Hybrid + rerank"
    semantic_only = mode == "Semantic only"
    # A fresh JobMatcher per click: BM25 + profile caches stay consistent with
    # the current index and sliders. The rebuild costs ~50 ms — negligible here.
    matcher = JobMatcher(
        rag=rag,
        semantic_weight=semantic_weight,
        reranker=get_reranker() if use_rerank else None,
        rerank=use_rerank,
    )

    try:
        with st.spinner("Matching..."):
            result = matcher.match(
                jd_text, k=k, apply_filters=apply_filters, semantic_only=semantic_only
            )
    except (RuntimeError, ValueError) as exc:
        st.error(f"{exc} — try 'Rebuild index' in the sidebar.")
        st.stop()

    query = result["query"]
    st.caption(
        f"**{query['title']}** · {len(query['required_skills'])} required skills · "
        f"{len(query['must_haves'])} must-haves · mode: `{query['mode']}` · "
        f"corpus: {corpus_label}"
    )

    matches = result["top_matches"]
    if not matches:
        st.info("No candidates survived the must-have filters — see below.")

    for rank, match in enumerate(matches, start=1):
        with st.container(border=True):
            head_left, head_right = st.columns([4, 1])
            with head_left:
                st.subheader(f"#{rank} {match['candidate_name']}")
                if match["matched_skills"]:
                    st.markdown(" ".join(f"`{s}`" for s in match["matched_skills"]))
                st.caption(match["reasoning"])
            with head_right:
                st.metric("Score", match["match_score"])
                st.caption(match["resume_path"])

            bars = st.columns(len(BREAKDOWN_BARS))
            breakdown = match.get("score_breakdown", {})
            for col, (key, label) in zip(bars, BREAKDOWN_BARS):
                value = clamp01(breakdown.get(key, 0.0))
                col.progress(value, text=f"{label}: {value:.2f}")

            with st.expander("Evidence excerpts"):
                for excerpt in match["relevant_excerpts"]:
                    st.markdown(f"- {bold_section_prefix(excerpt)}")

    filtered = result["filtered_out"]
    with st.expander(f"Filtered out ({len(filtered)})"):
        if not filtered:
            st.caption("Nobody was excluded by must-have requirements.")
        for entry in filtered:
            st.markdown(f"**{entry['candidate_name']}** — score {entry['match_score']}")
            for reason in entry["failed_requirements"]:
                st.markdown(f"  - {reason}")

    lat = result["latency_ms"]
    st.caption(
        f"semantic {lat['semantic_search']}ms · keyword {lat['keyword_search']}ms · "
        f"rerank {lat['rerank']}ms · total {lat['total']}ms"
    )
