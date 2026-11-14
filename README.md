
# MedVerse AI
### Multimodal Clinical Decision Support Platform

MedVerse AI is an academic project that explores how medical images, clinical report text, and structured patient information can be processed to generate supporting clinical evidence.

The project combines a web-based application with AI/ML experimentation involving medical image analysis, clinical text processing, OCR, and multimodal analysis.

**Important:** MedVerse AI is a research and academic prototype. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 1. Project Overview

Medical information is often distributed across different sources, including medical images, laboratory reports, clinical notes, and patient records.

MedVerse AI explores a workflow for bringing these sources together:

1. Patient information is recorded through the application.
2. Medical reports can be uploaded for text extraction.
3. Extracted clinical information can be processed to identify relevant findings.
4. Medical image analysis and clinical text analysis are explored through separate AI/ML components.
5. The platform provides APIs for retrieving AI evidence and performing available analysis operations.

The broader research direction is a confidence-aware multimodal clinical decision-support workflow that considers information from different modalities and aims to identify potential agreement or conflicts between them.

---

## 2. System Architecture

The project is organized into three main areas:

| Component | Technology / Purpose |
|---|---|
| Frontend | React, Vite |
| Backend | Python, FastAPI |
| Database | SQLite, SQLAlchemy |
| Authentication | JWT-based authentication |
| Medical report processing | OCR and clinical text extraction |
| AI/ML research | Python-based image and clinical text experiments |
| Image processing | Medical image classification experiments |
| Development | Git, GitHub, VS Code |

### High-level workflow

```text
                User
                 |
                 v
          React Frontend
                 |
                 v
           FastAPI Backend
                 |
       +---------+----------+
       |         |          |
       v         v          v
  Authentication Patients  Reports
       |         |          |
       +---------+----------+
                 |
                 v
             SQLite DB
                 |
                 v
          AI / Evidence APIs
                 |
       +---------+----------+
       |         |          |
       v         v          v
     OCR    Clinical NLP   X-ray
                           Analysis
                 |
                 v
        Evidence / Analysis
```

The AI/ML research code is maintained separately under `ai-services/`. The presence of a research module does not necessarily mean that every experiment or trained model is connected to the running web application.

---

## 3. Implemented Application Components

### 3.1 Backend API

The backend is developed using FastAPI and provides API endpoints for application operations.

The backend includes:

- User registration and login.
- JWT-based authentication.
- Patient information management.
- Medical report upload and storage.
- OCR and clinical information extraction workflows.
- AI evidence retrieval.
- X-ray and fusion analysis API routes.
- RAG question-answering API route.
- AI service status endpoint.

### 3.2 Database

The application uses **SQLite with SQLAlchemy** for database operations.

The database models include:

- `User` – user account information.
- `Patient` – patient details, symptoms, history, age, and gender.
- `MedicalReport` – uploaded report information and extracted OCR text.
- `AIEvidence` – evidence records associated with analysis.

The database is configured using the `DATABASE_URL` environment variable.

Default configuration:

```env
DATABASE_URL=sqlite:///./medverse.db
```

No PostgreSQL database is required by this project.

### 3.3 Authentication

The backend includes registration and login functionality with JWT-based authentication.

The secret key and token expiration settings are configurable through environment variables.

### 3.4 Medical Report Processing

The application provides report upload and processing workflows.

The report-processing components are intended to support:

- Medical report upload.
- OCR-based text extraction.
- Extraction of relevant clinical information.
- Storage and retrieval of report-related information.
- AI evidence generation and retrieval through available APIs.

Tesseract OCR is used for OCR processing where configured.

### 3.5 AI and Evidence APIs

The backend exposes API routes for available AI operations, including:

- AI service status.
- X-ray analysis.
- Multimodal fusion analysis.
- Clinical evidence retrieval.
- RAG-based question answering.

The availability and output of an analysis depend on its implementation, required data, and model artifacts being present.

---

## 4. AI/ML Research Components

The `ai-services/` directory contains AI/ML development and experimentation separate from the core backend application.

The research work explores the following areas.

### 4.1 Chest X-ray Analysis

Medical image classification experiments use convolutional neural networks, including DenseNet121.

The work explores classification of chest X-ray images and evaluation using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Trained model checkpoints and datasets are maintained separately and are not included in this repository.

### 4.2 Clinical Text Processing

Clinical text experiments explore extracting and representing relevant information from medical text.

The work includes clinical text processing and machine-learning approaches for deriving evidence from structured or unstructured clinical information.

### 4.3 Multimodal Analysis

The research direction explores combining information from different modalities, particularly:

- Medical images.
- Clinical report text.
- Structured clinical information.

The objective is to investigate how evidence from different sources can be combined and how potential conflicts between modalities can be identified.

### 4.4 Retrieval-Augmented Generation

The project also contains a RAG-related backend route and research components for retrieving relevant information to support question answering.

RAG outputs should be treated as supporting information, not as verified medical conclusions.

**Research implementation note:** Individual models, training scripts, experiment outputs, and backend API integrations are separate parts of the project. Do not assume every research experiment is deployed as a production-ready service.

---

## 5. Repository Structure

```text
MedVerse-AI/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   ├── routers/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── ...
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── ai-services/
│   ├── clinical/
│   ├── ocr/
│   ├── xray/
│   └── ...
│
├── data/
│   └── Dataset files are maintained locally.
│
├── docs/
├── figures/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── DATASETS.md
└── README.md
```

Some generated files, local databases, datasets, model checkpoints, and environment-specific files are intentionally excluded from version control.

---

## 6. Requirements

Before running the project, install:

- Python 3.10 or a compatible Python 3 version.
- Node.js and npm.
- Git.
- Tesseract OCR, if using the OCR functionality.

Some AI/ML experiments may require additional dependencies and model files beyond those needed to run the web application.

---

## 7. Setup and Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/pranavyabaskar-cmyk/MedVerse-AI.git
cd MedVerse-AI
```

### Step 2: Create and activate a Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\activate.bat
```

### Step 3: Install backend dependencies

From the repository root:

```bash
pip install -r backend/requirements.txt
```

If you are setting up the standalone AI/ML experiments, check their individual dependency requirements as well.

### Step 4: Configure environment variables

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder secret with a securely generated random value.

Example configuration:

```env
DATABASE_URL=sqlite:///./medverse.db
SECRET_KEY=replace-with-a-secure-random-secret
ACCESS_TOKEN_EXPIRE_MINUTES=60
TESSERACT_CMD=
UPLOAD_DIR=uploads
```

If Tesseract is installed in a custom location, configure `TESSERACT_CMD` with the executable path.

For example:

```env
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

Do not commit `.env` or place API keys, passwords, JWT secrets, or other credentials in source files.

### Step 5: Run the backend

From the repository root:

```bash
uvicorn backend.app.main:app --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

FastAPI interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

If your local application package setup requires running from the backend directory instead, use:

```powershell
cd backend
uvicorn app.main:app --reload
```

### Step 6: Run the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in the terminal.

---

## 8. API Routes

The backend contains the following API route groups:

| Route | Purpose |
|---|---|
| `/auth/register` | User registration |
| `/auth/login` | User login |
| `/patients` | Patient operations |
| `/reports/upload` | Medical report upload and persistence |
| `/evidence/{patient_id}` | Retrieve patient evidence |
| `/ai/fusion/{patient_id}` | Fusion analysis endpoint |
| `/ai/xray/{patient_id}` | X-ray analysis endpoint |
| `/ai/xray/upload` | X-ray upload and analysis workflow |
| `/ai/report/upload` | Report analysis workflow |
| `/ai/evidence/{patient_id}` | AI evidence retrieval |
| `/ai/status` | AI service status |
| `/ai/rag/ask` | RAG question-answering endpoint |

For exact HTTP methods, request schemas, response formats, and authentication requirements, refer to the running FastAPI documentation at `/docs`.

Note: `/reports/upload` and `/ai/report/upload` serve different workflows. The former is the report persistence workflow; the latter is an AI report-analysis workflow.

---

## 9. Running Tests

From the repository root:

```bash
python -m pytest backend/tests -v
```

Tests may require the backend dependencies and the corresponding application configuration to be installed.

---

## 10. Datasets and Model Artifacts

The project uses local datasets and trained model artifacts for AI/ML development.

These files are not included in the GitHub repository because of their size, dataset access conditions, and the need to manage research data separately.

Refer to [DATASETS.md](DATASETS.md) for:

- Dataset organization.
- Local data requirements.
- Model artifact requirements.
- Data preparation guidance.
- Reproducibility considerations.

You must obtain and configure the required datasets and model files separately before running experiments that depend on them.

---

## 11. Current Scope and Limitations

MedVerse AI is an academic research prototype. Its scope includes backend application development, patient and report workflows, OCR, clinical information processing, and AI/ML experimentation.

Important limitations:

- Model performance depends on the dataset, preprocessing, training configuration, and evaluation split.
- Some AI/ML components are experimental and may not be integrated into the application.
- Model checkpoints and datasets are not distributed with the repository.
- Results from experimental models should not be interpreted as clinical validation.
- The application is not intended for real-world diagnosis or autonomous treatment decisions.
- Reproducing all experiments requires obtaining the corresponding datasets, dependencies, and model artifacts.

---

## 12. Future Work

Potential areas for further development include:

- Improved multimodal feature fusion.
- Confidence estimation for individual modalities.
- Detection of conflicting clinical evidence.
- Explainability for image-based predictions.
- Better integration of clinical text and image analysis.
- More extensive evaluation on appropriately sourced datasets.
- Improved reproducibility and model deployment.

---

## 13. Disclaimer

MedVerse AI is developed for academic and research purposes only.

It is not a certified medical device and must not be used as the sole basis for diagnosis, treatment, or other clinical decisions.

All AI-generated outputs require independent review by qualified healthcare professionals.

---

## 14. Repository

GitHub: https://github.com/pranavyabaskar-cmyk/MedVerse-AI
