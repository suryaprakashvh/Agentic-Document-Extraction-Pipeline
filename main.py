import os
import json
import re
from typing import Dict, Any
from jsonschema import validate
from mistralai import Mistral
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OCR_MODEL = "mistral-ocr-latest"
LLM_MODEL = "llama-3.1-8b-instant"

CONFIDENCE_THRESHOLD = 0.75

# =========================
# CLIENTS
# =========================
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# =========================
# JSON SCHEMA (NULL-SAFE)
# =========================
ENROLLMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "first_name": {"type": ["string", "null"]},
        "last_name": {"type": ["string", "null"]},
        "date_of_birth": {"type": ["string", "null"]},
        "gender": {"type": ["string", "null"]},
        "email": {"type": ["string", "null"]},
        "address": {"type": ["string", "null"]},
        "city": {"type": ["string", "null"]},
        "state": {"type": ["string", "null"]},
        "postal_code": {"type": ["string", "null"]},
        "alternate_contact": {"type": ["string", "null"]},
        "alternate_phone": {"type": ["string", "null"]},
        "household_size": {"type": ["number", "null"]},
        "annual_income": {"type": ["number", "null"]}
    },
    "required": ["first_name", "last_name", "date_of_birth", "email"]
}

# =========================
# UTILITY — SAFE JSON EXTRACTION
# =========================
def extract_json_from_text(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM output")
    return json.loads(match.group())

# =========================
# AGENT 1 — OCR
# =========================
def ocr_agent(file_path: str) -> str:
    print("[OCR] Uploading file to Mistral...")
    with open(file_path, "rb") as f:
        uploaded_file = mistral_client.files.upload(
            file={"file_name": os.path.basename(file_path), "content": f},
            purpose="ocr"
        )

    print("[OCR] Running Mistral OCR 3...")
    response = mistral_client.ocr.process(
        model=OCR_MODEL,
        document={"type": "file", "file_id": uploaded_file.id}
    )

    return "\n\n".join(page.markdown for page in response.pages)

# =========================
# GROQ LLM CALL
# =========================
def groq_llm(system_prompt: str, user_text: str, temperature=0) -> str:
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# =========================
# AGENT 2 — CLASSIFIER
# =========================
def classify_document(text: str) -> str:
    print("[Classifier] Classifying document...")
    prompt = """
You are a document classifier.
Return ONLY one label:
- enrollment_form
- invoice
- unknown
"""
    return groq_llm(prompt, text[:4000])

# =========================
# AGENT 3 — EXTRACTION (NULL-SAFE)
# =========================
def extraction_agent(text: str, retries=2) -> Dict[str, Any]:
    print("[Extraction] Extracting structured data...")
    prompt = f"""
You are a strict data extraction engine.

Extract data strictly using this JSON schema:
{json.dumps(ENROLLMENT_SCHEMA, indent=2)}

Rules:
- Output ONE JSON object only
- No markdown
- No explanation
- Missing values → null
"""
    for attempt in range(retries + 1):
        try:
            llm_output = groq_llm(prompt, text)
            extracted = extract_json_from_text(llm_output)
            # Make all missing fields explicitly null
            for key in ENROLLMENT_SCHEMA["properties"].keys():
                if key not in extracted or extracted[key] in ("", None):
                    extracted[key] = None
            return extracted
        except Exception:
            print(f"[Extraction] Retry {attempt + 1} due to malformed JSON")
            if attempt == retries:
                raise

# =========================
# AGENT 4 — VALIDATION
# =========================
def validate_agent(data: Dict[str, Any]) -> None:
    print("[Validation] Validating schema...")
    validate(instance=data, schema=ENROLLMENT_SCHEMA)

# =========================
# AGENT 5 — CONFIDENCE (LLM-SCORING, FIXED)
# =========================
def confidence_agent_llm(extracted: Dict[str, Any], text: str) -> Dict[str, Dict]:
    print("[Confidence] Asking LLM to score fields...")
    prompt = f"""
You are an assistant that assigns a confidence score (0.0 to 1.0) to each extracted field
based on how sure you are that the value is correct from the OCR text.

OCR Text:
{text[:4000]}

Extracted JSON:
{json.dumps(extracted, indent=2)}

Output JSON format:
{{
  "first_name": {{"value": <value>, "confidence": <0-1>}},
  "last_name": {{"value": <value>, "confidence": <0-1>}},
  ...
}}

Rules:
- Confidence should reflect your certainty about the correctness of the value.
- Keep all keys from the extracted JSON.
- Missing values → null, confidence → 0.0
- Output valid JSON only.
"""

    # Attempt to get JSON from LLM
    try:
        llm_output = groq_llm(prompt, text)
        scored = extract_json_from_text(llm_output)
        if not isinstance(scored, dict):
            scored = {}
    except Exception as e:
        print(f"[Confidence] LLM failed, falling back to default scores: {e}")
        scored = {}

    # Ensure all keys exist
    for key, value in extracted.items():
        if key not in scored or not isinstance(scored[key], dict):
            scored[key] = {
                "value": value,
                "confidence": 0.0 if value is None else 1.0
            }
        else:
            # Ensure both 'value' and 'confidence' exist
            scored[key]["value"] = scored[key].get("value", value)
            scored[key]["confidence"] = scored[key].get("confidence", 0.0 if value is None else 1.0)

    return scored


# =========================
# AGENT 6 — NORMALIZATION
# =========================
def normalize_agent(scored: Dict[str, Dict]) -> Dict[str, Dict]:
    print("[Normalization] Normalizing fields...")
    for field, obj in scored.items():
        if field == "email" and obj["value"]:
            obj["value"] = obj["value"].lower()
        if field == "postal_code" and obj["value"]:
            obj["value"] = re.sub(r"\s+", "", obj["value"])
    return scored

# =========================
# AGENT 7 — CONSISTENCY
# =========================
def consistency_agent(scored: Dict[str, Dict]) -> None:
    print("[Consistency] Checking logic...")
    dob = scored.get("date_of_birth", {}).get("value", "")
    if dob and not re.match(r"\d{2}/\d{2}/\d{2,4}", dob):
        raise ValueError("Invalid DOB format")

# =========================
# RETRY CONTROLLER
# =========================
def needs_retry(scored: Dict[str, Dict]) -> bool:
    return any(v["confidence"] < CONFIDENCE_THRESHOLD for v in scored.values())

# =========================
# ORCHESTRATOR
# =========================
def run_pipeline(file_path: str) -> Dict[str, Any]:
    text = ocr_agent(file_path)

    doc_type = classify_document(text)
    print(f"[Router] Document type: {doc_type}")

    if doc_type != "enrollment_form":
        raise RuntimeError("Unsupported document type")

    extracted = extraction_agent(text)
    validate_agent(extracted)

    scored = confidence_agent_llm(extracted, text)

    if needs_retry(scored):
        print("[Retry] Low confidence detected — retrying extraction...")
        extracted = extraction_agent(text)
        validate_agent(extracted)
        scored = confidence_agent_llm(extracted, text)

    scored = normalize_agent(scored)
    consistency_agent(scored)

    return scored

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    result = run_pipeline("form.pdf")
    print(json.dumps(result, indent=2))
