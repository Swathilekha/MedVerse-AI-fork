from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from datetime import datetime
from ..database import Base

class MedicalReport(Base):
    __tablename__ = "medical_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    ocr_text = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
