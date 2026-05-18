"""
spaCy-based clinical entity extractor (Stage 1 of the two-stage NLP pipeline).

Uses an EntityRuler with hand-crafted patterns for five clinical entity types:
  DIAGNOSIS, MEDICATION, LAB_VALUE, PROCEDURE, CLINICAL_FINDING

Runs fast rule-based recognition so the Streamlit UI can display spaCy results
alongside the Claude API results (Stage 2) for comparison.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

DIAGNOSIS_PHRASES = [
    "type 2 diabetes mellitus",
    "type 2 diabetes",
    "type 1 diabetes mellitus",
    "type 1 diabetes",
    "diabetes mellitus",
    "diabetes",
    "hypertension",
    "essential hypertension",
    "stage 3 chronic kidney disease",
    "stage 4 chronic kidney disease",
    "stage 5 chronic kidney disease",
    "chronic kidney disease",
    "CKD",
    "COPD",
    "chronic obstructive pulmonary disease",
    "heart failure",
    "congestive heart failure",
    "acute decompensated heart failure",
    "atrial fibrillation",
    "AFib",
    "hyperlipidemia",
    "dyslipidemia",
    "coronary artery disease",
    "CAD",
    "asthma",
    "depression",
    "major depressive disorder",
    "pneumonia",
    "urinary tract infection",
    "UTI",
    "obesity",
    "morbid obesity",
    "hypothyroidism",
    "hyperthyroidism",
    "stroke",
    "ischemic stroke",
    "hemorrhagic stroke",
    "sepsis",
    "deep vein thrombosis",
    "DVT",
    "pulmonary embolism",
    "acute myocardial infarction",
    "myocardial infarction",
    "MI",
    "anemia",
    "iron deficiency anemia",
    "osteoporosis",
    "osteoarthritis",
    "rheumatoid arthritis",
    "chronic pain",
    "anxiety",
    "generalized anxiety disorder",
    "GERD",
    "gastroesophageal reflux disease",
]

MEDICATION_PHRASES = [
    "metformin",
    "lisinopril",
    "atorvastatin",
    "furosemide",
    "carvedilol",
    "apixaban",
    "semaglutide",
    "tiotropium",
    "budesonide",
    "formoterol",
    "prednisone",
    "prednisolone",
    "azithromycin",
    "losartan",
    "amlodipine",
    "ketorolac",
    "acetaminophen",
    "sertraline",
    "rosuvastatin",
    "metoprolol",
    "metoprolol succinate",
    "metoprolol tartrate",
    "nitroglycerin",
    "aspirin",
    "ferrous sulfate",
    "insulin",
    "insulin glargine",
    "insulin lispro",
    "levothyroxine",
    "omeprazole",
    "pantoprazole",
    "clopidogrel",
    "warfarin",
    "rivaroxaban",
    "spironolactone",
    "amoxicillin",
    "doxycycline",
    "albuterol",
    "salbutamol",
    "fluticasone",
    "gabapentin",
    "pregabalin",
    "hydrochlorothiazide",
    "bisoprolol",
    "ramipril",
    "enalapril",
    "valsartan",
    "empagliflozin",
    "dapagliflozin",
    "sitagliptin",
]

LAB_VALUE_PHRASES = [
    "HbA1c",
    "hemoglobin A1c",
    "hemoglobin A 1c",
    "BP",
    "blood pressure",
    "eGFR",
    "estimated GFR",
    "creatinine",
    "serum creatinine",
    "troponin",
    "troponin I",
    "troponin T",
    "BNP",
    "NT-proBNP",
    "SpO2",
    "oxygen saturation",
    "WBC",
    "white blood cell",
    "white blood cell count",
    "hemoglobin",
    "hematocrit",
    "FEV1",
    "LDL",
    "LDL cholesterol",
    "HDL",
    "HDL cholesterol",
    "total cholesterol",
    "triglycerides",
    "potassium",
    "sodium",
    "PTH",
    "parathyroid hormone",
    "TSH",
    "thyroid stimulating hormone",
    "albumin",
    "INR",
    "prothrombin time",
    "platelet count",
    "platelets",
    "glucose",
    "fasting glucose",
    "A1C",
]

PROCEDURE_PHRASES = [
    "echocardiogram",
    "transthoracic echocardiogram",
    "transesophageal echocardiogram",
    "chest X-ray",
    "chest x-ray",
    "chest xray",
    "ECG",
    "EKG",
    "12-lead ECG",
    "12 lead ECG",
    "electrocardiogram",
    "laparoscopic cholecystectomy",
    "cholecystectomy",
    "fundoscopic exam",
    "fundoscopy",
    "ophthalmoscopy",
    "nebulization",
    "nebulizer treatment",
    "colonoscopy",
    "mammography",
    "mammogram",
    "spirometry",
    "incentive spirometry",
    "CT scan",
    "CT chest",
    "CT abdomen",
    "MRI",
    "MRI brain",
    "ultrasound",
    "renal ultrasound",
    "Doppler ultrasound",
    "stress test",
    "nuclear stress test",
    "cardiac catheterization",
    "catheterization",
    "dialysis",
    "hemodialysis",
    "peritoneal dialysis",
    "biopsy",
    "renal biopsy",
    "bronchoscopy",
    "PET scan",
    "bone density scan",
    "DEXA scan",
]

CLINICAL_FINDING_PHRASES = [
    "chest pain",
    "shortness of breath",
    "dyspnea",
    "dyspnea on exertion",
    "edema",
    "bilateral leg swelling",
    "leg swelling",
    "lower extremity edema",
    "pitting edema",
    "crackles",
    "bibasilar crackles",
    "wheezes",
    "wheeze",
    "wheezing",
    "cough",
    "productive cough",
    "chronic cough",
    "nausea",
    "vomiting",
    "nausea and vomiting",
    "fever",
    "fatigue",
    "hemoptysis",
    "polyuria",
    "polydipsia",
    "polyphagia",
    "hypoglycemia",
    "hyperglycemia",
    "palpitations",
    "syncope",
    "dizziness",
    "headache",
    "diaphoresis",
    "night sweats",
    "weight loss",
    "weight gain",
    "anorexia",
    "abdominal pain",
    "flank pain",
    "back pain",
    "hematuria",
    "dysuria",
    "constipation",
    "diarrhea",
    "melena",
    "confusion",
    "altered mental status",
    "weakness",
    "numbness",
    "tingling",
    "blurred vision",
    "decreased appetite",
    "tachycardia",
    "bradycardia",
    "hypertensive",
]

# ---------------------------------------------------------------------------
# NLP pipeline builder (cached so the model is only loaded once)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_nlp():
    """Load spaCy model and attach EntityRuler. Cached after first call."""
    import spacy

    nlp = spacy.load("en_core_web_sm")

    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})

    patterns = []

    # Phrase patterns — the ruler accepts nlp.make_doc for multi-token phrases
    for phrase in DIAGNOSIS_PHRASES:
        patterns.append({"label": "DIAGNOSIS", "pattern": phrase})

    for phrase in MEDICATION_PHRASES:
        patterns.append({"label": "MEDICATION", "pattern": phrase})

    for phrase in LAB_VALUE_PHRASES:
        patterns.append({"label": "LAB_VALUE", "pattern": phrase})

    for phrase in PROCEDURE_PHRASES:
        patterns.append({"label": "PROCEDURE", "pattern": phrase})

    for phrase in CLINICAL_FINDING_PHRASES:
        patterns.append({"label": "CLINICAL_FINDING", "pattern": phrase})

    # Token-level regex patterns for lab values that appear as abbreviations
    # followed by a numeric reading, e.g. "HbA1c 7.8%" or "eGFR 42"
    lab_abbrev_patterns = [
        (r"(?i)HbA1c", "HbA1c"),
        (r"(?i)A1C", "A1C"),
        (r"(?i)eGFR", "eGFR"),
        (r"(?i)BNP", "BNP"),
        (r"(?i)NT-?proBNP", "NT-proBNP"),
        (r"(?i)SpO2", "SpO2"),
        (r"(?i)WBC", "WBC"),
        (r"(?i)FEV1", "FEV1"),
        (r"(?i)LDL", "LDL"),
        (r"(?i)HDL", "HDL"),
        (r"(?i)PTH", "PTH"),
        (r"(?i)TSH", "TSH"),
        (r"(?i)INR", "INR"),
    ]
    for regex, label_text in lab_abbrev_patterns:
        patterns.append(
            {
                "label": "LAB_VALUE",
                "pattern": [{"TEXT": {"REGEX": regex}}],
            }
        )

    ruler.add_patterns(patterns)
    return nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> list[dict]:
    """
    Run the spaCy rule-based entity extractor on *text*.

    Returns a list of dicts, each with keys:
        text       — the matched surface form
        label      — one of DIAGNOSIS, MEDICATION, LAB_VALUE,
                     PROCEDURE, CLINICAL_FINDING
        start_char — character offset of span start
        end_char   — character offset of span end

    Returns [] if *text* is empty/None or if the spaCy model is unavailable.
    """
    if not text or not text.strip():
        return []

    try:
        nlp = _build_nlp()
    except OSError as exc:
        logger.warning(
            "spaCy model 'en_core_web_sm' not available — entity extraction skipped. "
            "Install with: python -m spacy download en_core_web_sm\n"
            "Original error: %s",
            exc,
        )
        return []

    doc = nlp(text)

    clinical_labels = {
        "DIAGNOSIS",
        "MEDICATION",
        "LAB_VALUE",
        "PROCEDURE",
        "CLINICAL_FINDING",
    }

    results = []
    for ent in doc.ents:
        if ent.label_ in clinical_labels:
            results.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

    return results
