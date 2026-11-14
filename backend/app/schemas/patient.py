from pydantic import BaseModel

class PatientCreate(BaseModel):
    age: int
    gender: str
    medical_history: str = ""
    symptoms: str = ""

class PatientResponse(PatientCreate):
    id: int

    class Config:
        from_attributes = True
