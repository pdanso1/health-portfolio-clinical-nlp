"""
ICD-10 code mapper for clinical NLP.

Provides local lookup and verification of ICD-10 codes for diagnoses.
Maps multiple text forms of diagnoses to canonical ICD-10 codes.
Includes enrichment function to verify Claude parser's diagnosis list.

Public API
----------
lookup_icd10(diagnosis_text: str) -> tuple[str, str]
    Look up an ICD-10 code and description. Returns (code, description) or ("", "").

enrich_diagnoses(diagnoses: list[dict]) -> list[dict]
    Enrich a list of diagnoses with verified ICD-10 codes from local map.
"""

# ---------------------------------------------------------------------------
# ICD-10 Mapping Dictionary
# Maps multiple text forms to (code, description) tuples
# ---------------------------------------------------------------------------

ICD10_MAP = {
    # Type 2 Diabetes Mellitus (E11.9)
    "type 2 diabetes mellitus": ("E11.9", "Type 2 diabetes mellitus, unspecified"),
    "type 2 diabetes": ("E11.9", "Type 2 diabetes mellitus, unspecified"),
    "type 2 dm": ("E11.9", "Type 2 diabetes mellitus, unspecified"),
    "t2dm": ("E11.9", "Type 2 diabetes mellitus, unspecified"),
    "diabetes mellitus type 2": ("E11.9", "Type 2 diabetes mellitus, unspecified"),
    "diabetes type 2": ("E11.9", "Type 2 diabetes mellitus, unspecified"),
    "t2d": ("E11.9", "Type 2 diabetes mellitus, unspecified"),

    # Essential Hypertension (I10)
    "essential hypertension": ("I10", "Essential (primary) hypertension"),
    "hypertension": ("I10", "Essential (primary) hypertension"),
    "htn": ("I10", "Essential (primary) hypertension"),
    "primary hypertension": ("I10", "Essential (primary) hypertension"),

    # Chronic Kidney Disease Stage 3 (N18.3)
    "stage 3 chronic kidney disease": ("N18.3", "Chronic kidney disease, stage 3a/3b"),
    "ckd stage 3": ("N18.3", "Chronic kidney disease, stage 3a/3b"),
    "chronic kidney disease stage 3": ("N18.3", "Chronic kidney disease, stage 3a/3b"),
    "ckd 3": ("N18.3", "Chronic kidney disease, stage 3a/3b"),

    # Chronic Kidney Disease Stage 4 (N18.4)
    "stage 4 chronic kidney disease": ("N18.4", "Chronic kidney disease, stage 4"),
    "ckd stage 4": ("N18.4", "Chronic kidney disease, stage 4"),
    "chronic kidney disease stage 4": ("N18.4", "Chronic kidney disease, stage 4"),
    "ckd 4": ("N18.4", "Chronic kidney disease, stage 4"),

    # Chronic Kidney Disease Stage 5 (N18.5)
    "stage 5 chronic kidney disease": ("N18.5", "Chronic kidney disease, stage 5"),
    "ckd stage 5": ("N18.5", "Chronic kidney disease, stage 5"),
    "chronic kidney disease stage 5": ("N18.5", "Chronic kidney disease, stage 5"),
    "ckd 5": ("N18.5", "Chronic kidney disease, stage 5"),
    "end stage renal disease": ("N18.5", "Chronic kidney disease, stage 5"),
    "esrd": ("N18.5", "Chronic kidney disease, stage 5"),

    # COPD (J44.1)
    "copd": ("J44.1", "Unspecified chronic obstructive pulmonary disease"),
    "chronic obstructive pulmonary disease": ("J44.1", "Unspecified chronic obstructive pulmonary disease"),
    "obstructive airway disease": ("J44.1", "Unspecified chronic obstructive pulmonary disease"),

    # Heart Failure (I50.9)
    "heart failure": ("I50.9", "Unspecified systolic or diastolic heart failure"),
    "acute decompensated heart failure": ("I50.9", "Unspecified systolic or diastolic heart failure"),
    "congestive heart failure": ("I50.9", "Unspecified systolic or diastolic heart failure"),
    "chf": ("I50.9", "Unspecified systolic or diastolic heart failure"),
    "acute heart failure": ("I50.9", "Unspecified systolic or diastolic heart failure"),
    "hf": ("I50.9", "Unspecified systolic or diastolic heart failure"),

    # Asthma (J45.909)
    "asthma": ("J45.909", "Unspecified asthma with (acute) exacerbation"),
    "asthma with exacerbation": ("J45.909", "Unspecified asthma with (acute) exacerbation"),

    # Obesity (E66.9)
    "obesity": ("E66.9", "Obesity, unspecified"),
    "obese": ("E66.9", "Obesity, unspecified"),

    # Depression (F32.9)
    "depression": ("F32.9", "Major depressive disorder, single episode, unspecified"),
    "major depressive disorder": ("F32.9", "Major depressive disorder, single episode, unspecified"),
    "mdd": ("F32.9", "Major depressive disorder, single episode, unspecified"),

    # Atrial Fibrillation (I48.91)
    "atrial fibrillation": ("I48.91", "Unspecified atrial fibrillation"),
    "afib": ("I48.91", "Unspecified atrial fibrillation"),
    "a-fib": ("I48.91", "Unspecified atrial fibrillation"),
    "atrialfibrillation": ("I48.91", "Unspecified atrial fibrillation"),

    # Hyperlipidemia / Dyslipidemia (E78.5)
    "hyperlipidemia": ("E78.5", "Hyperlipidemia, unspecified"),
    "dyslipidemia": ("E78.5", "Hyperlipidemia, unspecified"),
    "elevated cholesterol": ("E78.5", "Hyperlipidemia, unspecified"),

    # Hypothyroidism (E03.9)
    "hypothyroidism": ("E03.9", "Hypothyroidism, unspecified"),
    "underactive thyroid": ("E03.9", "Hypothyroidism, unspecified"),

    # Coronary Artery Disease (I25.10)
    "coronary artery disease": ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
    "cad": ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
    "ischemic heart disease": ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),

    # Acute Myocardial Infarction (I21.9)
    "acute myocardial infarction": ("I21.9", "ST elevation myocardial infarction of unspecified site"),
    "acute mi": ("I21.9", "ST elevation myocardial infarction of unspecified site"),
    "myocardial infarction": ("I21.9", "ST elevation myocardial infarction of unspecified site"),
    "heart attack": ("I21.9", "ST elevation myocardial infarction of unspecified site"),

    # Stroke / Cerebrovascular Accident (I63.9)
    "stroke": ("I63.9", "Cerebral infarction, unspecified"),
    "ischemic stroke": ("I63.9", "Cerebral infarction, unspecified"),
    "cva": ("I63.9", "Cerebral infarction, unspecified"),

    # Pneumonia (J18.9)
    "pneumonia": ("J18.9", "Pneumonia, unspecified organism"),
    "bacterial pneumonia": ("J18.9", "Pneumonia, unspecified organism"),
    "viral pneumonia": ("J18.9", "Pneumonia, unspecified organism"),

    # Deep Vein Thrombosis (I82.409)
    "deep vein thrombosis": ("I82.409", "Acute embolism and thrombosis of deep veins of lower extremity, bilateral"),
    "dvt": ("I82.409", "Acute embolism and thrombosis of deep veins of lower extremity, bilateral"),
    "thrombophlebitis": ("I82.409", "Acute embolism and thrombosis of deep veins of lower extremity, bilateral"),

    # Sepsis (A41.9)
    "sepsis": ("A41.9", "Sepsis, unspecified organism"),
    "septic": ("A41.9", "Sepsis, unspecified organism"),
    "bacterial sepsis": ("A41.9", "Sepsis, unspecified organism"),

    # Urinary Tract Infection (N39.0)
    "urinary tract infection": ("N39.0", "Urinary tract infection, site not specified"),
    "uti": ("N39.0", "Urinary tract infection, site not specified"),
    "bladder infection": ("N39.0", "Urinary tract infection, site not specified"),
    "cystitis": ("N39.0", "Urinary tract infection, site not specified"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup_icd10(diagnosis_text: str) -> tuple[str, str]:
    """
    Look up an ICD-10 code and description for a diagnosis text.

    Performs case-insensitive and whitespace-normalized lookup against the
    local ICD10_MAP. First attempts exact match; then tries partial matching
    (substring containment).

    Parameters
    ----------
    diagnosis_text : str
        The diagnosis text to look up (e.g., "Type 2 Diabetes", "t2dm").

    Returns
    -------
    tuple[str, str]
        (code, description) if found, or ("", "") if not found.
        code is a valid ICD-10 code (e.g., "E11.9").
        description is the full text description.
    """
    if not diagnosis_text:
        return ("", "")

    # Normalize: lowercase and strip whitespace
    normalized_input = diagnosis_text.lower().strip()

    # First: exact match
    if normalized_input in ICD10_MAP:
        return ICD10_MAP[normalized_input]

    # Second: partial matching
    # Check if any key is contained in the input or vice versa
    for key, value in ICD10_MAP.items():
        if key in normalized_input or normalized_input in key:
            return value

    return ("", "")


def enrich_diagnoses(diagnoses: list[dict]) -> list[dict]:
    """
    Enrich a list of diagnoses with verified ICD-10 codes from the local map.

    For each diagnosis in the input list, attempts to look up the 'text' field
    in the local ICD10_MAP. If found, overwrites icd10_code and icd10_description
    with the verified values. If not found, keeps the original code (may be empty).

    Does not mutate the input list; returns a new list of copies.

    Parameters
    ----------
    diagnoses : list[dict]
        List of diagnosis dicts from Claude parser. Each dict should have keys:
        text, icd10_code, icd10_description, status, confidence.

    Returns
    -------
    list[dict]
        A new list of diagnosis dicts with verified ICD-10 codes.
    """
    enriched = []

    for diagnosis in diagnoses:
        # Create a copy to avoid mutating the original
        enriched_diagnosis = dict(diagnosis)

        # Try to look up the diagnosis text
        diagnosis_text = diagnosis.get("text", "")
        if diagnosis_text:
            code, description = lookup_icd10(diagnosis_text)
            if code:  # Only update if we found a match
                enriched_diagnosis["icd10_code"] = code
                enriched_diagnosis["icd10_description"] = description

        enriched.append(enriched_diagnosis)

    return enriched
