from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime
from datetime import datetime
from ..database import Base

class AIEvidence(Base):
    __tablename__ = "ai_evidence"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    source = Column(Text, nullable=False)
    data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
