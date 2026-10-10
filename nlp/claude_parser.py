"""
Claude API clinical entity parser (Stage 2 of the two-stage NLP pipeline).

Uses claude-haiku-5-5 to extract structured clinical entities that
rule-based spaCy may miss: ICD-10/LOINC/CPT codes, negation detection,
confidence scores, and temporal context.

Public API
----------
parse_note(note_text: str) -> dict
    Returns a dict with five keys (diagnoses, medications, lab_values,
    procedures, clinical_findings), each mapping to a list of entity dicts.
    On any error, returns the same structure with empty lists.
"""

import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 8000  # Haiku 5.5 thinking counts toward max_tokens

EMPTY_RESULT: dict = {
    "diagnoses": [],
    "medications": [],
    "lab_values": [],
    "procedures": [],
    "clinical_findings": [],
}

_PROMPT_TEMPLATE = """\
You are a clinical NLP system. Extract structured clinical entities from this note.

Return a JSON object with these keys:
- diagnoses: [{{text, icd10_code, icd10_description, status: "active"|"historical"|"negated", confidence}}]
- medications: [{{name, dose, frequency, route, confidence}}]
- lab_values: [{{test_name, loinc_code, value, units, interpretation: "normal"|"abnormal"|"critical", confidence}}]
- procedures: [{{name, cpt_code_if_known, date_if_mentioned, confidence}}]
- clinical_findings: [{{finding, status: "present"|"absent"|"uncertain", confidence}}]

Clinical note:
{note_text}"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_note(note_text: str) -> dict:
    """
    Call the Claude API to extract structured clinical entities from *note_text*.

    Parameters
    ----------
    note_text : str
        Raw clinical note text.

    Returns
    -------
    dict
        {
            "diagnoses": [...],
            "medications": [...],
            "lab_values": [...],
            "procedures": [...],
            "clinical_findings": [...]
        }
        On any error (missing API key, API failure, JSON parse error),
        returns the same structure with empty lists.
    """
    if not note_text or not note_text.strip():
        return dict(EMPTY_RESULT)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY is not set — Claude parsing skipped. "
            "Set the environment variable or configure it via Streamlit secrets."
        )
        return dict(EMPTY_RESULT)

    prompt_text = _PROMPT_TEMPLATE.format(note_text=note_text)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )

        # Extract the text content from the response
        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text = block.text
                break

        if not raw_text:
            logger.warning("Claude returned an empty response for the clinical note.")
            return dict(EMPTY_RESULT)

        return _parse_json_response(raw_text)

    except anthropic.AuthenticationError:
        logger.error(
            "ANTHROPIC_API_KEY is invalid or revoked. "
            "Check the key in your environment or Streamlit secrets."
        )
        return dict(EMPTY_RESULT)
    except anthropic.RateLimitError:
        logger.warning("Claude API rate limit hit — try again shortly.")
        return dict(EMPTY_RESULT)
    except anthropic.APIStatusError as exc:
        logger.error("Claude API error %s: %s", exc.status_code, exc.message)
        return dict(EMPTY_RESULT)
    except anthropic.APIConnectionError:
        logger.error("Network error connecting to the Claude API.")
        return dict(EMPTY_RESULT)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error calling Claude API: %s", exc)
        return dict(EMPTY_RESULT)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"diagnoses", "medications", "lab_values", "procedures", "clinical_findings"}


def _parse_json_response(raw_text: str) -> dict:
    """
    Extract and validate the JSON object from Claude's text response.

    Claude typically returns a bare JSON object, but may occasionally wrap it
    in markdown code fences (```json ... ```). We handle both cases.

    Parameters
    ----------
    raw_text : str
        Raw text returned by Claude.

    Returns
    -------
    dict
        Parsed result with the five required keys. Missing keys are filled
        with empty lists. On parse failure, returns EMPTY_RESULT.
    """
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line (e.g. ```json) and closing fence
        inner_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse Claude response as JSON: %s\nRaw text (first 500 chars): %.500s",
            exc,
            raw_text,
        )
        return dict(EMPTY_RESULT)

    if not isinstance(data, dict):
        logger.error(
            "Claude response parsed as %s, expected dict. Raw: %.500s",
            type(data).__name__,
            raw_text,
        )
        return dict(EMPTY_RESULT)

    # Ensure all required keys are present; fill missing ones with []
    result = dict(EMPTY_RESULT)
    for key in _REQUIRED_KEYS:
        value = data.get(key)
        result[key] = value if isinstance(value, list) else []

    return result
