# Agentic Document Extraction Pipeline

## Overview

This project implements an **agent-based document understanding pipeline** designed to extract **structured, schema-driven JSON** from scanned or handwritten documents.

The system combines:

- **Mistral OCR 3** for high-quality OCR
- **Groq-hosted LLMs (LLaMA 3.1)** for classification, extraction, and reasoning
- **JSON Schema validation** to enforce strict output contracts
- **Confidence-driven retries** to improve extraction reliability

The pipeline is intentionally lightweight and modular, making it suitable for research, experimentation, and controlled production workflows.

---

## Key Capabilities

- OCR for scanned and handwritten documents
- Document type classification
- Schema-constrained structured extraction
- Null-safe output for missing data
- Field-level confidence scoring
- Automatic retries on low-confidence output
- Post-processing normalization and consistency checks

---

## High-Level Architecture

```
Document (PDF / Image)
        ↓
     OCR Agent
        ↓
 Document Classifier
        ↓
 Structured Extraction
        ↓
 Schema Validation
        ↓
 Confidence Scoring
        ↓
 Normalization & Checks
        ↓
 Final JSON Output
```

---

## How It Works

1. **OCR Agent**
   The input document is uploaded to Mistral OCR and converted into structured markdown text.

2. **Document Classification**
   A lightweight LLM classifier determines whether the document matches the expected processing flow.

3. **Structured Extraction**
   An LLM extracts information strictly according to a predefined JSON Schema.
   Any missing or uncertain values are explicitly returned as `null`.

4. **Schema Validation**
   The extracted output is validated using `jsonschema` to ensure structural correctness.

5. **Confidence Scoring**
   A secondary LLM pass assigns confidence scores to each extracted value based on OCR evidence.

6. **Retry Logic**
   If confidence falls below a configurable threshold, extraction is retried automatically.

7. **Normalization & Consistency Checks**
   Extracted values are normalized and validated for basic logical consistency.

---

## Design Principles

- **Schema-first extraction**
- **Fail-safe defaults** (missing data never breaks the pipeline)
- **Deterministic orchestration**
- **Minimal orchestration overhead**
- **LLM-assisted reasoning, not blind OCR output**

---

## Prerequisites

- Python **3.9+**
- API access to:

  - Mistral AI
  - Groq

---

## Installation

```bash
pip install mistralai groq python-dotenv jsonschema
```

---

## Configuration

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Pipeline

```bash
python sample.py
```

The pipeline processes a document file path defined in the entry point and returns a **validated, confidence-scored JSON object**.

---

## Output Format

The final output is a **pure JSON object** where:

- Each key represents a schema-defined attribute
- Each value contains:

  - the extracted value (or `null`)
  - a confidence score between `0.0` and `1.0`

Example structure:

```json
{
  "some_field": {
    "value": "example",
    "confidence": 0.91
  }
}
```

---

## Retry Strategy

A retry is triggered when **any confidence score falls below the configured threshold**.
Retries are limited and controlled to prevent infinite loops.

---
