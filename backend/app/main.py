from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .models.all_models import User, Patient, MedicalReport, AIEvidence
from .routers import auth, patients, reports, evidence, ai


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="MedVerse AI",
    description="Multimodal Clinical Decision Support Platform",
    version="0.2.0"
)


# React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(reports.router)
app.include_router(evidence.router)
app.include_router(ai.router)


@app.get("/")
def root():
    return {
        "project": "MedVerse AI",
        "status": "running",
        "stage": "Multimodal Clinical Decision Support"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}