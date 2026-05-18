"""
Clinical NLP Parser — Streamlit app.

Two-panel layout:
  Left  (~1/3): note input, sample selector, extract button, pipeline metrics
  Right (~2/3): entity highlighting + tabbed Claude parser results
"""

import os

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical NLP Parser",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# API key handling — before any API call
# ---------------------------------------------------------------------------
api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
if not api_key:
    st.warning(
        "⚠️ ANTHROPIC_API_KEY not set. Claude parsing will return empty results. "
        "Set it in .streamlit/secrets.toml or environment."
    )
else:
    os.environ["ANTHROPIC_API_KEY"] = api_key  # ensure claude_parser picks it up

# ---------------------------------------------------------------------------
# NLP module imports
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, os.path.dirname(__file__))

from nlp.sample_notes import SAMPLE_NOTES
from nlp.extractor import extract_entities
from nlp.claude_parser import parse_note
from nlp.icd_mapper import enrich_diagnoses

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

ENTITY_COLORS = {
    "DIAGNOSIS": "#FFA500",        # amber/orange
    "MEDICATION": "#6495ED",       # cornflower blue
    "LAB_VALUE": "#90EE90",        # light green
    "PROCEDURE": "#DDA0DD",        # plum/purple
    "CLINICAL_FINDING": "#FFB6C1", # light pink/red
}

ENTITY_TEXT_COLORS = {
    "DIAGNOSIS": "black",
    "MEDICATION": "white",
    "LAB_VALUE": "black",
    "PROCEDURE": "black",
    "CLINICAL_FINDING": "black",
}


def build_highlighted_html(text: str, entities: list[dict]) -> str:
    """
    Build HTML string with <mark> tags around entity spans.

    Processes spans in start_char order; overlapping spans are skipped.
    """
    if not entities:
        # Return plain text in a pre-formatted block
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<div style="white-space: pre-wrap; font-family: monospace; font-size: 0.9em;">{escaped}</div>'

    # Sort by start position; on tie prefer longer span
    sorted_ents = sorted(entities, key=lambda e: (e["start_char"], -e["end_char"]))

    html_parts = []
    cursor = 0

    for ent in sorted_ents:
        start = ent["start_char"]
        end = ent["end_char"]
        label = ent["label"]

        # Skip spans that overlap with already-consumed text
        if start < cursor:
            continue

        # Plain text before this entity
        plain = text[cursor:start]
        html_parts.append(
            plain.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

        bg_color = ENTITY_COLORS.get(label, "#FFFF00")
        text_color = ENTITY_TEXT_COLORS.get(label, "black")
        span_text = text[start:end].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        html_parts.append(
            f'<mark style="background-color: {bg_color}; color: {text_color}; '
            f'padding: 2px 4px; border-radius: 3px;" title="{label}">'
            f'{span_text}</mark>'
        )
        cursor = end

    # Remaining text after last entity
    tail = text[cursor:]
    html_parts.append(
        tail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )

    inner = "".join(html_parts)
    return (
        '<div style="white-space: pre-wrap; font-family: monospace; font-size: 0.9em; '
        'line-height: 1.6;">'
        + inner
        + "</div>"
    )


def render_color_legend() -> None:
    """Render a color legend row for entity types."""
    legend_items = [
        ("DIAGNOSIS",       ENTITY_COLORS["DIAGNOSIS"],       "Diagnosis"),
        ("MEDICATION",      ENTITY_COLORS["MEDICATION"],       "Medication"),
        ("LAB_VALUE",       ENTITY_COLORS["LAB_VALUE"],        "Lab Value"),
        ("PROCEDURE",       ENTITY_COLORS["PROCEDURE"],        "Procedure"),
        ("CLINICAL_FINDING",ENTITY_COLORS["CLINICAL_FINDING"], "Finding"),
    ]
    badges = []
    for label, color, display in legend_items:
        text_color = ENTITY_TEXT_COLORS.get(label, "black")
        badges.append(
            f'<span style="background-color: {color}; color: {text_color}; '
            f'padding: 3px 10px; border-radius: 12px; font-size: 0.85em; '
            f'margin-right: 8px;">{display}</span>'
        )
    st.markdown(" ".join(badges), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1, 2])

# ============================================================
# LEFT COLUMN — Input
# ============================================================
with left_col:
    st.title("🔬 Clinical NLP Parser")
    st.markdown(
        "Parse clinical notes with a two-stage pipeline: "
        "**spaCy** rule-based entity recognition + **Claude** structured extraction."
    )

    st.divider()

    # Sample note selector
    sample_options = ["Custom note"] + list(SAMPLE_NOTES.keys())
    selected_sample = st.selectbox("Select a sample note", options=sample_options)

    if selected_sample == "Custom note":
        note_text = st.text_area(
            "Paste clinical note:",
            height=300,
            placeholder="Paste a clinical note here...",
        )
    else:
        note_text = st.text_area(
            "Note text (editable):",
            value=SAMPLE_NOTES[selected_sample],
            height=300,
        )

    extract_button = st.button("Extract Entities", type="primary", use_container_width=True)

    # ============================================================
    # Processing — on button click
    # ============================================================
    if extract_button:
        if not note_text or not note_text.strip():
            st.warning("Please enter or select a clinical note before extracting.")
        else:
            with st.spinner("Extracting entities…"):
                # Stage 1: spaCy
                spacy_entities = extract_entities(note_text)

                # Stage 2: Claude
                claude_result = parse_note(note_text)

                # Stage 3: Enrich diagnoses with local ICD-10 map
                claude_result["diagnoses"] = enrich_diagnoses(claude_result["diagnoses"])

                # Store in session state
                st.session_state["results"] = {
                    "note_text": note_text,
                    "spacy_entities": spacy_entities,
                    "claude_result": claude_result,
                }

            # Safe here: guarded by `if extract_button`, so next pass won't re-trigger processing.
            st.rerun()

    # Pipeline metrics (shown after extraction)
    if "results" in st.session_state:
        results = st.session_state["results"]
        spacy_count = len(results["spacy_entities"])

        claude_result = results["claude_result"]
        claude_count = sum(
            len(claude_result.get(k, []))
            for k in ("diagnoses", "medications", "lab_values", "procedures", "clinical_findings")
        )

        st.divider()
        st.markdown("**Pipeline summary**")
        m1, m2 = st.columns(2)
        m1.metric("spaCy entities", spacy_count)
        m2.metric("Claude entities", claude_count)

# ============================================================
# RIGHT COLUMN — Results
# ============================================================
with right_col:
    if "results" not in st.session_state:
        st.info("Select a note on the left and click **Extract Entities** to begin.")
    else:
        results = st.session_state["results"]
        note_text_result = results["note_text"]
        spacy_entities = results["spacy_entities"]
        claude_result = results["claude_result"]

        # --------------------------------------------------------
        # Entity Highlighting
        # --------------------------------------------------------
        st.subheader("Entity Highlighting")

        highlighted_html = build_highlighted_html(note_text_result, spacy_entities)
        st.markdown(
            f'<div style="border: 1px solid #ddd; border-radius: 6px; '
            f'padding: 12px; background-color: #fafafa; max-height: 340px; '
            f'overflow-y: auto;">'
            f'{highlighted_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Legend:**")
        render_color_legend()

        st.divider()

        # --------------------------------------------------------
        # Tabbed Claude parser results
        # --------------------------------------------------------
        st.subheader("Structured Extraction (Claude)")

        tab_dx, tab_med, tab_lab, tab_proc, tab_find = st.tabs(
            ["🏥 Diagnoses", "💊 Medications", "🧪 Lab Values", "⚕️ Procedures", "🔍 Findings"]
        )

        # --- Diagnoses tab ---
        with tab_dx:
            diagnoses = claude_result.get("diagnoses", [])
            if not diagnoses:
                st.info("No diagnoses extracted.")
            else:
                rows = []
                for d in diagnoses:
                    rows.append({
                        "Text": d.get("text", ""),
                        "ICD-10 Code": d.get("icd10_code", ""),
                        "Description": d.get("icd10_description", ""),
                        "Status": d.get("status", ""),
                        "Confidence": round(float(d.get("confidence", 0.0)), 2),
                    })
                st.dataframe(rows, use_container_width=True)

        # --- Medications tab ---
        with tab_med:
            medications = claude_result.get("medications", [])
            if not medications:
                st.info("No medications extracted.")
            else:
                rows = []
                for m in medications:
                    rows.append({
                        "Name": m.get("name", ""),
                        "Dose": m.get("dose", ""),
                        "Frequency": m.get("frequency", ""),
                        "Route": m.get("route", ""),
                        "Confidence": round(float(m.get("confidence", 0.0)), 2),
                    })
                st.dataframe(rows, use_container_width=True)

        # --- Lab Values tab ---
        with tab_lab:
            lab_values = claude_result.get("lab_values", [])
            if not lab_values:
                st.info("No lab values extracted.")
            else:
                rows = []
                for lv in lab_values:
                    rows.append({
                        "Test": lv.get("test_name", ""),
                        "LOINC Code": lv.get("loinc_code", ""),
                        "Value": lv.get("value", ""),
                        "Units": lv.get("units", ""),
                        "Interpretation": lv.get("interpretation", ""),
                        "Confidence": round(float(lv.get("confidence", 0.0)), 2),
                    })
                st.dataframe(rows, use_container_width=True)

        # --- Procedures tab ---
        with tab_proc:
            procedures = claude_result.get("procedures", [])
            if not procedures:
                st.info("No procedures extracted.")
            else:
                rows = []
                for p in procedures:
                    rows.append({
                        "Procedure": p.get("name", ""),
                        "CPT Code": p.get("cpt_code_if_known", ""),
                        "Date Mentioned": p.get("date_if_mentioned", ""),
                        "Confidence": round(float(p.get("confidence", 0.0)), 2),
                    })
                st.dataframe(rows, use_container_width=True)

        # --- Clinical Findings tab ---
        with tab_find:
            findings = claude_result.get("clinical_findings", [])
            if not findings:
                st.info("No clinical findings extracted.")
            else:
                rows = []
                for f in findings:
                    rows.append({
                        "Finding": f.get("finding", ""),
                        "Status": f.get("status", ""),
                        "Confidence": round(float(f.get("confidence", 0.0)), 2),
                    })
                st.dataframe(rows, use_container_width=True)
