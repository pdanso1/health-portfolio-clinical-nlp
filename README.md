# Clinical NLP Parser

A two-stage clinical entity extraction pipeline combining rule-based NLP with LLM-powered structured parsing.

## Live Demo

https://pdanso1-health-portfolio-clinical-nlp.streamlit.app

---

## Clinical Context

### The Problem: Unstructured Data in EHRs

Roughly 80% of clinically meaningful data in electronic health records exists as unstructured free text — progress notes, discharge summaries, consultation letters, nursing assessments. This information is rich with diagnostic, therapeutic, and outcome data, but it is largely invisible to analytics systems without natural language processing.

Manual abstraction (chart review by trained coders or nurses) is the traditional workaround, but it is slow, expensive, inconsistent, and does not scale to population-level analytics. A single HEDIS or Stars measure abstraction campaign for a 100,000-member health plan can require thousands of hours of manual review.

Clinical NLP is the technology layer that bridges unstructured documentation and structured analytics.

### What Clinical NLP Does

- Converts free-text clinical notes into structured, queryable data elements
- Tags entities (diagnoses, medications, lab values, procedures, clinical findings) with standardized codes (ICD-10, LOINC, CPT)
- Detects negation ("no chest pain"), uncertainty ("possible pneumonia"), and temporal context ("history of MI")
- Enables downstream analytics: quality measure abstraction, risk stratification, care gap identification, prior authorization support

### The Two-Stage Approach

**Stage 1 — Rule-based extraction (spaCy EntityRuler)**

The spaCy pipeline uses a hand-crafted pattern library with 239 clinical terms across five entity types. It runs in milliseconds, is fully deterministic, and has high precision for known terminology. It catches what we know to look for.

**Stage 2 — LLM extraction (Claude API)**

Claude (claude-haiku-4-5-20251001) receives the full note and returns a structured JSON object containing diagnoses with ICD-10 codes, medications with dose/frequency/route, lab values with LOINC codes and interpretations, procedures with CPT codes, and clinical findings with presence/absence status. It handles ambiguous phrasing, novel terminology, and complex linguistic structures that rules cannot anticipate.

**Why two stages?** Rules are fast, interpretable, and audit-friendly — critical in regulated healthcare environments. LLMs are flexible but non-deterministic, slower, and API-dependent. Combining them provides both the precision of rules and the coverage of language models. In production systems, this hybrid approach is the industry standard (see: AWS Comprehend Medical + custom rules, Google Healthcare NLP API + custom postprocessors).

### Why Negation Detection Matters

The most famous failure mode in clinical NLP: extracting "chest pain" from the phrase "no chest pain." In production, false positives from missed negation cause:

- **Quality measure miscalculation:** A patient flagged with "chest pain" may incorrectly populate a cardiac quality measure denominator
- **Incorrect risk scores:** Conditions the patient does not have inflate risk scores, leading to wrong outreach priority
- **Alert fatigue:** Clinical decision support alerts fire on negated findings, training clinicians to ignore alerts

This app explicitly marks negated findings with `status: "negated"` (diagnoses) or `status: "absent"` (clinical findings), keeping them visible for audit while excluding them from positive-finding counts.

### Why Temporal Context Matters

"History of diabetes" (resolved or historical) and "newly diagnosed diabetes" (active) are the same string of words but completely different clinical realities. Conflating them causes:

- Incorrect denominator selection for quality measures (e.g., CDC-HbA1c applies to active diabetes patients)
- Inflated chronic condition counts in population health registries
- Incorrect care gap generation (ordering A1c testing for a patient whose diabetes was resolved)

This app captures temporal context via the `status` field on diagnoses: `"active"`, `"historical"`, or `"negated"`.

### Real-World Applications

| Use Case | How Clinical NLP Helps |
|---|---|
| **HEDIS / Stars measure abstraction** | Automate chart review to identify numerator hits from unstructured notes |
| **Prior authorization** | Extract documented clinical criteria from notes to support auth requests |
| **Care gap identification** | Find patients missing recommended screenings mentioned in notes but not coded |
| **Population health cohort building** | Tag members by condition, medication class, or care event from notes |
| **Risk stratification** | Surface active conditions and clinical findings for risk model features |
| **Utilization review** | Extract procedure and diagnosis context for concurrent review decisions |

---

## Technical Architecture

```
Clinical Note (free text)
        │
        ▼
┌─────────────────────────┐
│  Stage 1: spaCy         │  239 clinical patterns across 5 entity types
│  EntityRuler            │  Deterministic, sub-millisecond, high precision
└──────────┬──────────────┘
           │  entity spans (text, label, char offsets)
           ▼
┌─────────────────────────┐
│  Stage 2: Claude API    │  claude-haiku-4-5-20251001
│  (claude-haiku)         │  Structured JSON: ICD-10, LOINC, CPT codes
│                         │  Negation + temporal context detection
└──────────┬──────────────┘
           │  diagnoses, medications, labs, procedures, findings
           ▼
┌─────────────────────────┐
│  ICD-10 Mapper          │  Local code lookup to validate/enrich diagnoses
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Streamlit UI           │  Entity highlighting + tabbed structured output
│                         │  Color-coded spans, pipeline metrics
└─────────────────────────┘
```

---

## Stack

| Component | Technology |
|---|---|
| UI framework | Streamlit |
| Rule-based NLP | spaCy (en_core_web_sm) |
| LLM extraction | Anthropic Claude API (claude-haiku-4-5-20251001) |
| Data display | pandas DataFrames |
| Language | Python 3.10+ |

---

## Setup

**Local development:**

```bash
git clone https://github.com/pdanso1/health-portfolio-clinical-nlp
cd health-portfolio-clinical-nlp
pip install -r requirements.txt
```

Add your API key to `.streamlit/secrets.toml`:

```toml
[secrets]
ANTHROPIC_API_KEY = "sk-ant-..."
```

Run the app:

```bash
streamlit run app.py
```

The spaCy model (`en_core_web_sm`) is installed automatically via pip from `requirements.txt`. No separate `python -m spacy download` step is needed.

---

## Streamlit Cloud Deployment

1. Fork or push this repo to GitHub
2. Connect the repo in [Streamlit Cloud](https://streamlit.io/cloud)
3. Set the API key in **App Settings → Secrets**:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

4. Deploy — Streamlit Cloud will install all dependencies from `requirements.txt`, including the spaCy model wheel

---

## Data Privacy

All sample notes bundled with the app are **synthetic** (not real patient data, no PHI). They were written to illustrate realistic clinical documentation patterns without referencing any real individuals.

Custom notes pasted by the user in the "Custom note" text area are sent to the Anthropic Claude API for entity extraction. **Do not paste real patient notes or any PHI into the custom input field.** This app is a portfolio demonstration tool and is not HIPAA-compliant.

---

## About

Built by **Paa Danso**, Medical Lab Scientist II, M.S. Biological Data Science. This project is part of a healthcare analytics portfolio demonstrating applied clinical informatics skills — bridging laboratory science, data engineering, and healthcare NLP.

Portfolio: [GitHub](https://github.com/pdanso1)
