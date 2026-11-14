
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.patient import Patient
from ..models.user import User
from ..schemas.patient import PatientCreate, PatientResponse
from ..utils.dependencies import get_current_user

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.post("", response_model=PatientResponse)
def create_patient(
    payload: PatientCreate,
    db: Session = Depends(get_db)
):
    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db)
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return patient


@router.get("/me")
def get_my_patient(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient access only")

    patient = db.query(Patient).filter(
        Patient.user_id == current_user.id
    ).first()

    return patient


@router.post("/me", response_model=PatientResponse)
def create_my_patient(
    payload: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient access only")

    existing = db.query(Patient).filter(
        Patient.user_id == current_user.id
    ).first()

    if existing:
        raise HTTPException(
            status_code=409,
            detail="A patient profile already exists for this account"
        )

    patient = Patient(
        user_id=current_user.id,
        age=payload.age,
        gender=payload.gender,
        medical_history=payload.medical_history,
        symptoms=payload.symptoms
    )

    db.add(patient)
    db.commit()
    db.refresh(patient)

    return patient
