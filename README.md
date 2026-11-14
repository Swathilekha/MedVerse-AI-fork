# MedVerse AI — Review 2 Starter

This starter implements the first ~30% Review-2 foundation:
- FastAPI backend
- SQLite by default for zero-setup local development (PostgreSQL-ready)
- User registration/login with JWT
- Patient creation/retrieval
- Medical report upload endpoint
- OCR service placeholder using Tesseract
- Clinical value extraction from OCR text
- AI evidence record
- Chest X-ray service placeholder

## Important
The X-ray model is intentionally a placeholder in this starter. Do not claim model accuracy until a real dataset/model has been integrated and evaluated.

## Quick start

### 1. Create and activate a virtual environment

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Run the API
From the project root:
```bash
uvicorn backend.app.main:app --reload
```

Open:
http://127.0.0.1:8000/docs

## PostgreSQL
For today's sprint, SQLite is used by default so the project runs immediately.
To use PostgreSQL, set DATABASE_URL in `.env`, for example:
DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/medverse

## Tesseract
Install the Tesseract OCR engine separately on Windows/Linux/macOS.
Then set TESSERACT_CMD in `.env` if the executable is not on PATH.

## Review-2 implementation target
1. Backend
2. Database
3. Authentication
4. Patient module
5. Report upload
6. OCR
7. Clinical information extraction
8. X-ray model integration/evaluation
9. AI evidence representation

Fusion, Clinical Reasoning Engine, XAI, RAG and LLM are intentionally left for later stages.
