import os, json, shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.patient import Patient
from ..models.report import MedicalReport
from ..models.evidence import AIEvidence
from ..config import UPLOAD_DIR
from ..services.ocr_service import extract_text
from ..services.clinical_extraction import extract_clinical_values

router = APIRouter(prefix="/reports", tags=["Medical Reports"])

@router.post("/upload")
def upload_report(
    patient_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=404, detail="Patient not found")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(file.filename or "report")
    path = os.path.join(UPLOAD_DIR, safe_name)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text(path)
    except Exception as exc:
        text = f"OCR unavailable: {exc}"

    extracted = extract_clinical_values(text)

    report = MedicalReport(
        patient_id=patient_id,
        file_name=safe_name,
        ocr_text=text
    )
    db.add(report)

    evidence = AIEvidence(
        patient_id=patient_id,
        source="medical_report_ocr",
        data=json.dumps(extracted)
    )
    db.add(evidence)
    db.commit()
    db.refresh(report)

    return {
        "report_id": report.id,
        "file_name": safe_name,
        "ocr_text": text,
        "clinical_values": extracted
    }
