import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.patient import Patient
from ..models.report import MedicalReport
from ..models.evidence import AIEvidence

router = APIRouter(prefix="/evidence", tags=["AI Evidence"])

@router.get("/{patient_id}")
def get_ai_evidence(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    reports = db.query(MedicalReport).filter(MedicalReport.patient_id == patient_id).all()
    evidence = db.query(AIEvidence).filter(AIEvidence.patient_id == patient_id).all()

    return {
        "patient": {
            "id": patient.id,
            "age": patient.age,
            "gender": patient.gender,
            "medical_history": patient.medical_history,
            "symptoms": patient.symptoms
        },
        "reports": [
            {"id": r.id, "file_name": r.file_name, "ocr_text": r.ocr_text}
            for r in reports
        ],
        "ai_evidence": [
            {"id": e.id, "source": e.source, "data": json.loads(e.data)}
            for e in evidence
        ]
    }
